import os
import random
import signal
import sys
import time
from collections import deque
from heapq import heappop, heappush

# секунд: завершаемся незадолго до SIGTERM (10 мин). Для локальных тестов
# бюджет можно задать переменной окружения FVS_TIME_BUDGET.
TIME_BUDGET = float(os.environ.get("FVS_TIME_BUDGET", "570.0"))

stop = False


def on_sigterm(signum, frame):
    global stop
    stop = True


def read_graph():
    """Возвращает n, adj (0-индексация) и обратный список pred."""
    data = sys.stdin.buffer.read().split(b"\n")
    n = int(data[0].split()[0])
    adj = []
    for line in data[1:n + 1]:
        if line.strip():
            adj.append([int(x) - 1 for x in line.split()])
        else:
            adj.append([])
    pred = [[] for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            pred[v].append(u)
    return n, adj, pred


def kahn_acyclic(n, adj, deleted):
    """True, если граф без вершин deleted ацикличен (проверка Кана)."""
    indeg = [0] * n
    for u in range(n):
        if deleted[u]:
            continue
        for v in adj[u]:
            if not deleted[v]:
                indeg[v] += 1
    q = deque(u for u in range(n) if not deleted[u] and indeg[u] == 0)
    cnt = 0
    while q:
        u = q.popleft()
        cnt += 1
        for v in adj[u]:
            if not deleted[v]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
    return cnt == sum(1 for u in range(n) if not deleted[u])


def construct(n, adj, pred, rng):
    """Одна жадная конструкция допустимого множества X (0-индексация)."""
    alive = [True] * n
    indeg = [len(pred[u]) for u in range(n)]
    outdeg = [len(adj[u]) for u in range(n)]

    zero = deque()
    for u in range(n):
        if indeg[u] == 0 or outdeg[u] == 0:
            zero.append(u)

    def remove(u):
        if not alive[u]:
            return
        alive[u] = False
        for w in adj[u]:
            if alive[w]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    zero.append(w)
                elif outdeg[w] > 0:
                    heappush(heap, (-(indeg[w] * outdeg[w]), w))
        for w in pred[u]:
            if alive[w]:
                outdeg[w] -= 1
                if outdeg[w] == 0:
                    zero.append(w)
                elif indeg[w] > 0:
                    heappush(heap, (-(indeg[w] * outdeg[w]), w))

    heap = []
    for u in range(n):
        heappush(heap, (-(indeg[u] * outdeg[u]), u))

    def top_alive():
        while heap:
            neg, u = heap[0]
            if alive[u] and -neg == indeg[u] * outdeg[u]:
                heappop(heap)
                return u
            heappop(heap)
        best, bv = -1, -1
        for u in range(n):
            if alive[u]:
                w = indeg[u] * outdeg[u]
                if w > bv:
                    bv, best = w, u
        return best

    X = []
    while zero:
        u = zero.popleft()
        remove(u)

    while True:
        if stop:
            break  # получил SIGTERM — выходим;
        start = top_alive()
        if start == -1:
            break
        # обход по исходящим дугам: после kernel каждая вершина имеет out>=1,
        # поэтому обязательно придём в уже посещённую вершину (найдём цикл)
        seen_pos = {}
        path = []
        cur = start
        while cur not in seen_pos:
            seen_pos[cur] = len(path)
            path.append(cur)
            nxt, nw = -1, -1
            for w in adj[cur]:
                if alive[w]:
                    wt = indeg[w] * outdeg[w]
                    if wt > nw:
                        nw, nxt = wt, w
            cur = nxt
        cycle = path[seen_pos[cur]:]

        if rng.random() < 0.9:
            v = cycle[0]
            wv = indeg[v] + outdeg[v]
            r = rng.random
            for u in cycle[1:]:
                wu = indeg[u] + outdeg[u]
                if wu > wv or (wu == wv and r() < 0.5):
                    v = u
                    wv = wu
        else:
            v = rng.choice(cycle)

        X.append(v)
        remove(v)
        while zero:
            u = zero.popleft()
            remove(u)

    return X


def add_safe(n, adj, pred, v, del_set, budget):
    """
    True, если вершину v можно вернуть из ответа в граф так,
    что граф останется ацикличным.
    Обход ограничен бюджетом шагов (budget — список [остаток]) и лимитом шагов
    на одного кандидата: при исчерпании считаем, что возвращать рискованно
    """
    cap = 500
    pset = set(pred[v])
    seen = set()
    stack = []
    for w in adj[v]:
        if w == v:
            return False
        if w in del_set:
            continue
        seen.add(w)
        stack.append(w)
    steps = 0
    while stack:
        u = stack.pop()
        steps += 1
        if steps > cap or budget[0] <= 0:
            return False
        budget[0] -= 1
        if u == v or u in pset:
            return False
        for w in adj[u]:
            if w == v or w in del_set:
                continue
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return True


def improve(n, adj, pred, X, rng):
    """
    Несколько проходов: возвращаем в граф вершины, без которых он ацикличен.
    Для очень больших X улучшение дорого и, пропускаем его,
    а в остальном ограничиваем суммарным бюджетом шагов.
    """
    if len(X) > 30000:
        return X
    del_set = set(X)
    cand = list(X)
    budget = [3_000_000]
    for _ in range(2):
        rng.shuffle(cand)
        new_cand = []
        for v in cand:
            if v not in del_set:
                continue
            if budget[0] <= 0:
                new_cand.append(v)
                continue
            if add_safe(n, adj, pred, v, del_set, budget):
                del_set.discard(v)
            else:
                new_cand.append(v)
        cand = new_cand
        if not cand or budget[0] <= 0:
            break
    return list(del_set)


def main():
    signal.signal(signal.SIGTERM, on_sigterm)
    n, adj, pred = read_graph()
    rng = random.Random()

    start_time = time.monotonic()
    best = None
    attempt = 0

    while not stop and time.monotonic() - start_time < TIME_BUDGET:
        attempt += 1
        rng.seed(attempt * 1000003 + 12345)

        X = construct(n, adj, pred, rng)
        X = improve(n, adj, pred, X, rng)

        deleted = [False] * n
        for v in X:
            deleted[v] = True
        if kahn_acyclic(n, adj, deleted):
            if best is None or len(X) < len(best):
                best = X

    if best is None:
        # не должно случиться (граф гарантированно с циклом, конструкция корректна),
        # но страховка: допустимый ответ — все вершины
        best = list(range(1, n + 1))
    else:
        best = [v + 1 for v in best]

    sys.stdout.write("".join(f"{v}\n" for v in best))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
