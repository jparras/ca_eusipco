import numpy as np
import matplotlib.pyplot as plt
from tikzplotlib import save
from copy import deepcopy


def ca(actions, u1, u2, delta, n_com, threat_action, threat_u,  pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma, pl=False):
    # Phase 1: obtain candidate for a_o using random sampling
    a_o = []
    nmp = 100  # Number of points to test (max)
    for c in range(n_com):
        # Search for Player 1
        search = True
        i = 0  # Number of points tests
        while search and i < nmp:
            candidate_a = np.array([actions[np.random.randint(low=0, high=len(actions))],
                                    actions[np.random.randint(low=0, high=len(actions))]])  # Generate two random actions
            u_candidate = payoff(candidate_a[0], candidate_a[1], pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma)
            ac_aux = deepcopy(actions)
            ac_aux.remove(candidate_a[0]) # All actions but current
            dev = np.amax(payoff(np.array(ac_aux), candidate_a[1], pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma)[0])
            if not (u_candidate[0] - (1-delta)*dev - delta*threat_u[0] < 0) and u_candidate[0] >= threat_u[0]:  # Candidate point found for P1
                search = False
                # Check if point is valid for player 2 (i.e., communicate!)
                ac_aux = deepcopy(actions)
                ac_aux.remove(candidate_a[1])  # All actions but current
                dev = np.amax(payoff(candidate_a[0], np.array(ac_aux), pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma)[1])
                if not (u_candidate[1] - (1 - delta) * dev - delta * threat_u[1] < 0) and u_candidate[1] >= threat_u[1]:  # Point valid for player 2
                    a_o.append(np.hstack([candidate_a, np.array(u_candidate)]))
            else:
                i += 1
        # Search for Player 2
        search = True
        i = 0  # Number of points tests
        while search and i < nmp:
            candidate_a = np.array([actions[np.random.randint(low=0, high=len(actions))],
                                    actions[np.random.randint(low=0, high=len(actions))]])
            u_candidate = payoff(candidate_a[0], candidate_a[1], pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma)
            ac_aux = deepcopy(actions)
            ac_aux.remove(candidate_a[1])  # All actions but current
            dev = np.amax(payoff(candidate_a[0], np.array(ac_aux), pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma)[1])
            if not(u_candidate[1] - (1 - delta) * dev - delta * threat_u[1] < 0) and u_candidate[1] >= threat_u[
                1]:  # Candidate point found for P2
                search = False
                # Check if point is valid for P1 (i.e., communicate!)
                ac_aux = deepcopy(actions)
                ac_aux.remove(candidate_a[0])  # All actions but current
                dev = np.amax(payoff(np.array(ac_aux), candidate_a[1], pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma)[0])
                if not (u_candidate[0] - (1 - delta) * dev - delta * threat_u[0] < 0) and u_candidate[0] >= threat_u[0]:  # Point valid for P1
                    a_o.append(np.hstack([candidate_a, np.array(u_candidate)]))
            else:
                i += 1
    a1, a2 = threat_action[0], threat_action[1]
    p1, p2 = threat_u[0], threat_u[1]
    # Erase NE from the valid grid (it is already a valid strategy)
    for i in reversed(range(len(a_o))):
        aux = a_o[i]
        if aux[0] == threat_action[0] and aux[1] == threat_action[1]:
            del a_o[i]
    if len(a_o) > 0:  # Pareto pruning
        p1_grid = np.array(a_o)
        p1_grid = p1_grid[np.argsort(p1_grid[:, 2])]  # Sort by P1 payoff value
        p2_grid = np.array(a_o)
        p2_grid = p2_grid[np.argsort(p2_grid[:, 3])]  # Sort by P2 payoff value

        search = True
        while search:
            # Run jountly controlled lottery
            w1 = np.random.rand(1)
            w2 = np.random.rand(1)
            w = w1 + w2
            if w > 1:
                w = w - 1
            id = int(np.floor(len(p1_grid) * w)[0])  # Action to start to erase dominated strategies
            a1, a2 = a_o[id][0], a_o[id][1]
            # P1 erases dominated strategies
            for i in range(len(p1_grid)):
                if p1_grid[i, 0] == a1 and p1_grid[i, 1] == a2:
                    break
            p1_grid = p1_grid[i:]
            # P2 erases dominated strategies
            for i in range(len(p2_grid)):
                if p2_grid[i, 0] == a1 and p2_grid[i, 1] == a2:
                    break
            p2_grid = p2_grid[i:]
            # Intersect to obtain a new valid grid
            new_grid = np.array([x for x in set(tuple(x) for x in p1_grid) & set(tuple(x) for x in p2_grid)])
            if len(new_grid) == 0:  # The valid strategy is the one already selected (a1, a2)
                search = False
            elif len(new_grid) == 1:  # Single value: return it as valid
                a1, a2 = new_grid[0, 0], new_grid[0, 1]
                p1, p2 = new_grid[0, 2], new_grid[0, 3]
                search = False
            else: # Reorder and repeat the process
                p1_grid = p1_grid[np.argsort(p1_grid[:, 2])]  # Sort by P1 payoff value
                p2_grid = p2_grid[np.argsort(p2_grid[:, 3])]  # Sort by P2 payoff value
    if pl:
        plt.plot(p1, p2, 'go', markersize=15)
        plt.plot(u1, u2, 'co', markersize=8)
        plt.plot(threat_u[0], threat_u[1], 'sr', markersize=10)
        if len(a_o) > 0:
            plt.plot(np.array(a_o)[:, 2], np.array(a_o)[:, 3], 'kx')
        save('payoff.tex')
        plt.show()

    return a1, a2, p1, p2


def baseline_learn(actions, u1, u2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma, nreps):
    a1m = actions[-1]
    a2m = actions[-1]
    a1v = []
    a2v = []
    p1_content = False
    p2_content = False
    c = 3
    max_payoff1 = np.amax(u1)
    min_payoff1 = np.amin(u1)
    max_payoff2 = np.amax(u2)
    min_payoff2 = np.amin(u2)

    for i in range(nreps):
        t = i + 1
        epsilon = 1 / np.sqrt(t)
        if p1_content:
            prob_vector = [(epsilon ** c) / (len(actions) - 1)] * len(actions)
            prob_vector[actions.index(a1m)] = 1 - epsilon ** c
            aindex = np.argmax(np.random.multinomial(1, np.array(prob_vector)))
            a1 = actions[aindex]
        else:
            prob_vector = [1 / len(actions)] * len(actions)
            aindex = np.argmax(np.random.multinomial(1, np.array(prob_vector)))
            a1 = actions[aindex]
        if p2_content:
            prob_vector = [(epsilon ** c) / (len(actions) - 1)] * len(actions)
            prob_vector[actions.index(a2m)] = 1 - epsilon ** c
            aindex = np.argmax(np.random.multinomial(1, np.array(prob_vector)))
            a2 = actions[aindex]
        else:
            prob_vector = [1 / len(actions)] * len(actions)
            aindex = np.argmax(np.random.multinomial(1, np.array(prob_vector)))
            a2 = actions[aindex]
        u1n, u2n = payoff(a1, a2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma)
        u1n = (u1n - min_payoff1) / (max_payoff1 - min_payoff1)
        u2n = (u2n - min_payoff2) / (max_payoff2 - min_payoff2)
        if (p1_content and a1 != a1m) or not p1_content:
            a1m = a1
            aindex = np.argmax(np.random.multinomial(1, np.array([epsilon ** (1 - u1n), 1 - epsilon ** (1 - u1n)])))
            if aindex < 0.5:
                p1_content = True
            else:
                p1_content = False
        if (p2_content and a2 != a2m) or not p2_content:
            a2m = a2
            aindex = np.argmax(np.random.multinomial(1, np.array([epsilon ** (1 - u2n), 1 - epsilon ** (1 - u2n)])))
            if aindex < 0.5:
                p2_content = True
            else:
                p2_content = False
        a1v.append(a1m)
        a2v.append(a2m)
    return a1m, a2m, a1v, a2v


def simulate(tmax, u1, u2, a1o, a2o, a1p, a2p, actions):
    # No deviation case
    rwd_nd = np.zeros((2, tmax))

    action1 = actions.index(a1o)
    action2 = actions.index(a2o)

    for t in range(tmax):
        rwd_nd[0, t] = u1[action1, action2]
        rwd_nd[1, t] = u2[action1, action2]

    # Simulation with deviation at t=tdev
    rwd_dev = np.zeros((2, tmax))
    tdev = 1
    for t in range(tmax):
        if t < tdev:
            action1 = actions.index(a1o)
            action2 = actions.index(a2o)
        elif t == tdev:
            action1 = actions.index(a1o)
            action2 = int(np.argmax(u2[action1, :]))
        else:  # Grim
            action1 = actions.index(a1p)
            action2 = actions.index(a2p)
        rwd_dev[0, t] = u1[action1, action2]
        rwd_dev[1, t] = u2[action1, action2]

    return rwd_nd, rwd_dev


def distance(p1, p2):  # Returns distance between two positions
    return np.sqrt(np.sum(np.square(p1-p2)))


def att(p1, p2, alpha=4):  # REturns the attenuation between two positions
    return distance(p1, p2) ** (-alpha)


def sinr(p1, p2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma):
    sinr1ul = p_ul * att(pos_sb1, pos_u1) / (
                n0 + p_ul * att(pos_sb1, pos_u2) + gamma * (p1 + p2 * att(pos_sb1, pos_sb2)))
    sinr1dl = p1 * att(pos_sb1, pos_u1) / (
                n0 + p2 * att(pos_u1, pos_sb2) + gamma * (p_ul + p_ul * att(pos_u2, pos_u1)))
    sinr2ul = p_ul * att(pos_sb2, pos_u2) / (
                n0 + p_ul * att(pos_sb2, pos_u1) + gamma * (p2 + p1 * att(pos_sb1, pos_sb2)))
    sinr2dl = p2 * att(pos_sb2, pos_u2) / (
                n0 + p1 * att(pos_u2, pos_sb1) + gamma * (p_ul + p_ul * att(pos_u2, pos_u1)))
    return sinr1ul, sinr1dl, sinr2ul, sinr2dl


def payoff(p1, p2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma):
    sinr1ul, sinr1dl, sinr2ul, sinr2dl = sinr(p1, p2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma)
    return np.log(sinr1ul) + np.log(sinr1dl), np.log(sinr2ul) + np.log(sinr2dl)


if __name__=="__main__":

    np.random.seed(10)  # For repeatability
    N = 2  # Number of small cells
    pos_sb1 = np.array([10, 10])  # Position of small base 1
    pos_sb2 = np.array([0, 0])  # Position of small base 2
    pos_u1 = np.array([1, 8])  # Position of user of sb1
    pos_u2 = np.array([5, 5])  # Position of user of sb2

    n0 = 0.001  # Thermal noise
    p_ul = 10  # Constant uplink power!
    gamma = 0.001  # Cochannel interference factor

    actions = [5, 10, 15, 20, 25, 30]  # Power levels (actions!)

    pv = []
    u1 = np.zeros((len(actions), len(actions)))
    u2 = np.zeros((len(actions), len(actions)))
    for p1 in actions:
        for p2 in actions:
            pv.append(payoff(p1, p2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma))
            u1[actions.index(p1), actions.index(p2)] = pv[-1][0]
            u2[actions.index(p1), actions.index(p2)] = pv[-1][1]

    # SIMULATION 1: DIFFERENT DELTA VALUES
    tmax = 500
    nrep = 50
    deltav = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    ncom = 30
    data = np.zeros((6, 2, len(deltav)))
    print('Simulation 1: different delta values')
    for i, delta in enumerate(deltav):
        print('Delta value ', i, ' of ', len(deltav))
        rwd_nd_bas = np.zeros((2, tmax))
        rwd_dev_bas = np.zeros((2, tmax))
        rwd_nd_ca = np.zeros((2, tmax))
        rwd_dev_ca = np.zeros((2, tmax))
        rwd_nd_ne = np.zeros((2, tmax))
        rwd_dev_ne = np.zeros((2, tmax))

        for rep in range(nrep):
            a1m, a2m, a1v, a2v = baseline_learn(actions, u1, u2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul,
                                                gamma, ncom)  # Learning phase baseline
            a1ca, a2ca, p1ca, p2ca = ca(actions, u1, u2, delta, ncom, np.array([actions[-1], actions[-1]]),
                                        np.array([u1[-1, -1], u2[-1, -1]]), pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul,
                                        gamma)

            nd, dev = simulate(tmax, u1, u2, a1m, a2m, actions[-1], actions[-1], actions)
            rwd_nd_bas = rwd_nd_bas + 1/nrep * nd
            rwd_dev_bas = rwd_dev_bas + 1 / nrep * dev
            nd, dev = simulate(tmax, u1, u2, a1ca, a2ca, actions[-1], actions[-1], actions)
            rwd_nd_ca = rwd_nd_ca + 1 / nrep * nd
            rwd_dev_ca = rwd_dev_ca + 1 / nrep * dev
            nd, dev = simulate(tmax, u1, u2, actions[-1], actions[-1], actions[-1], actions[-1], actions)
            rwd_nd_ne = rwd_nd_ne + 1 / nrep * nd
            rwd_dev_ne = rwd_dev_ne + 1 / nrep * dev

        # Save cumulative rewards
        data[0, 0, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_nd_bas[0, :])
        data[0, 1, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_nd_bas[1, :])
        data[1, 0, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_dev_bas[0, :])
        data[1, 1, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_dev_bas[1, :])
        data[2, 0, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_nd_ca[0, :])
        data[2, 1, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_nd_ca[1, :])
        data[3, 0, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_dev_ca[0, :])
        data[3, 1, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_dev_ca[1, :])
        data[4, 0, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_nd_ne[0, :])
        data[4, 1, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_nd_ne[1, :])
        data[5, 0, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_dev_ne[0, :])
        data[5, 1, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_dev_ne[1, :])

    plt.plot(deltav, data[0, 0, :] - data[4, 0, :], 'bo-', label='P1-ND-Bas')
    plt.plot(deltav, data[0, 1, :] - data[4, 1, :], 'bx-', label='P2-ND-Bas')
    plt.plot(deltav, data[0, 0, :] - data[4, 0, :] + data[0, 1, :] - data[4, 1, :], 'b:', label='Bas Agg')
    #plt.plot(deltav, data[1, 0, :] - data[5, 0, :], 'bo:', label='P1-D-Bas')
    #plt.plot(deltav, data[1, 1, :] - data[5, 1, :], 'bx:', label='P2-D-Bas')
    plt.plot(deltav, data[2, 0, :] - data[4, 0, :], 'ko-', label='P1-ND-CA')
    plt.plot(deltav, data[2, 1, :] - data[4, 1, :], 'kx-', label='P2-ND-CA')
    plt.plot(deltav, data[2, 0, :] - data[4, 0, :] + data[2, 1, :] - data[4, 1, :], 'k:', label='CA Agg')
    #plt.plot(deltav, data[3, 0, :] - data[5, 0, :], 'ko:', label='P1-D-CA')
    #plt.plot(deltav, data[3, 1, :] - data[5, 1, :], 'kx:', label='P2-D-CA')
    #plt.plot(deltav, np.zeros(len(deltav)), 'r')
    plt.legend(loc='best')
    plt.xlabel('Delta')
    plt.ylabel('Payoff gain')
    save('delta.tex')
    plt.show()

    # SIMULATION 2: DIFFERENT NCOM VALUES
    tmax = 500
    nrep = 50
    delta = 0.95
    ncomv = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    data = np.zeros((6, 2, len(ncomv)))
    print('Simulation 2: different com values')
    for i, ncom in enumerate(ncomv):
        print('Ncom value ', i, ' of ', len(ncomv))
        rwd_nd_bas = np.zeros((2, tmax))
        rwd_dev_bas = np.zeros((2, tmax))
        rwd_nd_ca = np.zeros((2, tmax))
        rwd_dev_ca = np.zeros((2, tmax))
        rwd_nd_ne = np.zeros((2, tmax))
        rwd_dev_ne = np.zeros((2, tmax))

        for rep in range(nrep):
            a1m, a2m, a1v, a2v = baseline_learn(actions, u1, u2, pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul,
                                                gamma, ncom)  # Learning phase baseline
            a1ca, a2ca, p1ca, p2ca = ca(actions, u1, u2, delta, ncom, np.array([actions[-1], actions[-1]]),
                                        np.array([u1[-1, -1], u2[-1, -1]]), pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul,
                                        gamma)

            nd, dev = simulate(tmax, u1, u2, a1m, a2m, actions[-1], actions[-1], actions)
            rwd_nd_bas = rwd_nd_bas + 1/nrep * nd
            rwd_dev_bas = rwd_dev_bas + 1 / nrep * dev
            nd, dev = simulate(tmax, u1, u2, a1ca, a2ca, actions[-1], actions[-1], actions)
            rwd_nd_ca = rwd_nd_ca + 1 / nrep * nd
            rwd_dev_ca = rwd_dev_ca + 1 / nrep * dev
            nd, dev = simulate(tmax, u1, u2, actions[-1], actions[-1], actions[-1], actions[-1], actions)
            rwd_nd_ne = rwd_nd_ne + 1 / nrep * nd
            rwd_dev_ne = rwd_dev_ne + 1 / nrep * dev

        # Save cumulative rewards
        data[0, 0, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_nd_bas[0, :])
        data[0, 1, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_nd_bas[1, :])
        data[1, 0, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_dev_bas[0, :])
        data[1, 1, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_dev_bas[1, :])
        data[2, 0, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_nd_ca[0, :])
        data[2, 1, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_nd_ca[1, :])
        data[3, 0, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_dev_ca[0, :])
        data[3, 1, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_dev_ca[1, :])
        data[4, 0, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_nd_ne[0, :])
        data[4, 1, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_nd_ne[1, :])
        data[5, 0, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_dev_ne[0, :])
        data[5, 1, i] = (1 - delta) * np.sum(np.power(delta, np.arange(tmax)) * rwd_dev_ne[1, :])

    plt.plot(ncomv, data[0, 0, :] - data[4, 0, :], 'bo-', label='P1-ND-Bas')
    plt.plot(ncomv, data[0, 1, :] - data[4, 1, :], 'bx-', label='P2-ND-Bas')
    plt.plot(ncomv, data[0, 0, :] - data[4, 0, :] + data[0, 1, :] - data[4, 1, :], 'b:', label='Bas Agg')
    plt.plot(ncomv, data[2, 0, :] - data[4, 0, :], 'ko-', label='P1-ND-CA')
    plt.plot(ncomv, data[2, 1, :] - data[4, 1, :], 'kx-', label='P2-ND-CA')
    plt.plot(ncomv, data[2, 0, :] - data[4, 0, :] + data[2, 1, :] - data[4, 1, :], 'k:', label='CA Agg')
    plt.legend(loc='best')
    plt.xlabel('Ncom')
    plt.ylabel('Payoff gain')
    save('ncom.tex')
    plt.show()

    # SIMULATION 3: PAYOFF REGION PLOT
    delta = 0.95
    ncom = 30
    _ = ca(actions, u1, u2, delta, ncom, np.array([actions[-1], actions[-1]]), np.array([u1[-1, -1], u2[-1, -1]]),
           pos_sb1, pos_sb2, pos_u1, pos_u2, n0, p_ul, gamma, pl=True)

    print('Done')