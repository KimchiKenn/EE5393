# EE5393 Homework 2 P2 
# Kenji Vang | vang3841@umn.edu
# Claude.ai helped generate code for this file

# This code will generate the following Biquad Filter CRN simulation
# The filter computes: Y[n] = (1/8)*A[n] + (1/8)*D1[n-1] + (1/8)*D2[n-1]
# where:
#    A[n]  = X[n] + (1/8)*D1[n-1] + (1/8)*D2[n-1]
 #   D1[n] = A[n]        (first delay — what RGB1 stores)
  #  D2[n] = D1[n-1]     (second delay — what RGB2 stores)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Rate constants
# kslow: slow reactions — signal transfers that
#        only happen when the right phase is active
# kfast: fast reactions — immediate equilibration,
#        indicator generation/consumption
# Sr/Sg/Sb: constant source concentrations for
#           absence indicator generation
# ─────────────────────────────────────────────
kslow = 1.0
kfast = 10.0
Sr    = 1.0   # source for r (red absence indicator)
Sg    = 1.0   # source for g (green absence indicator)
Sb    = 1.0   # source for b (blue absence indicator)
 
 
def biquad_odes(y, t):
    """
    ODE system for the biquad CRN.
    y is the vector of all species concentrations.
    t is time (not used explicitly — autonomous system).
 
    Each line corresponds to one or more reactions.
    We compute d[species]/dt by summing contributions
    from every reaction that produces or consumes it.
 
    Reaction rate = rate_constant * product_of_reactant_concentrations
    (mass-action kinetics)
    """
 
    # ── Unpack species ──────────────────────────────
    X   = max(y[0],  0)
    A   = max(y[1],  0)
    R1  = max(y[2],  0)
    G1  = max(y[3],  0)
    B1  = max(y[4],  0)
    C   = max(y[5],  0)
    F   = max(y[6],  0)
    R2  = max(y[7],  0)
    G2  = max(y[8],  0)
    B2  = max(y[9],  0)
    E   = max(y[10], 0)
    H   = max(y[11], 0)
    Yo  = max(y[12], 0)
    R1p = max(y[13], 0)
    G1p = max(y[14], 0)
    B1p = max(y[15], 0)
    R2p = max(y[16], 0)
    G2p = max(y[17], 0)
    B2p = max(y[18], 0)
    Rp  = max(y[19], 0)
    Bp  = max(y[20], 0)
    r   = max(y[21], 0)
    g   = max(y[22], 0)
    b   = max(y[23], 0)
 
    # ── Reaction rates (mass action) ────────────────
    # Group 1: fanout and scaling
    # "g + X -> A"
    # Why: X can only transfer to A when green phase is
    # active (g present), preventing premature transfer
    r_gX_A      = kslow * g * X
 
    # "8A -> Y"  means rate proportional to A^8 / 8!
    # We approximate this as a linear drain:
    # dY/dt += (1/8)*kfast*A  (linear scaling)
    # This is the standard CRN approximation for 1/8
    r_A_Y       = kfast * (1/8) * A
 
    # "A -> R1" (A feeds into first delay)
    r_A_R1      = kfast * A
 
    # "C -> Y" (1/8 scaling from first delay output)
    r_C_Y       = kfast * (1/8) * C
 
    # "F -> R2" (F feeds second delay)
    r_F_R2      = kfast * F
 
    # "E -> Y" (1/8 scaling from second delay output)
    r_E_Y       = kfast * (1/8) * E
 
    # "H -> A" (H feeds back, 1/8 scaling)
    r_H_A       = kfast * (1/8) * H
 
    # "F -> A" (F feeds back, 1/8 scaling)
    r_F_A       = kfast * (1/8) * F
 
    # "R1 -> C + F" (RGB1 outputs two copies)
    r_R1_CF     = kfast * R1
 
    # "R2 -> E + H" (RGB2 outputs two copies)
    r_R2_EH     = kfast * R2
 
    # Group 2: RGB phase transitions for delay 1
    # "b + R1 -> G1"
    # Why: R1 can only move to G1 when blue phase is
    # absent (b present), enforcing phase ordering
    r_bR1_G1    = kslow * b * R1
 
    # "r + G1 -> B1"
    # Why: G1 moves to B1 only in absence of red
    r_rG1_B1    = kslow * r * G1
 
    # "g + B1 -> Y"
    # Why: B1 contributes to output in absence of green
    r_gB1_Y     = kslow * g * B1
 
    # RGB phase transitions for delay 2
    r_bR2_G2    = kslow * b * R2
    r_rG2_B2    = kslow * r * G2
    r_gB2_Y     = kslow * g * B2
 
    # Group 3: color concentration indicators
    # "2R1 -> 2R1 + R1'"
    # Why: R1' tracks how much red-phase signal exists
    # so absence indicators know when red phase is done
    r_R1_R1p    = kfast * R1 * R1
    r_G1_G1p    = kfast * G1 * G1
    r_B1_B1p    = kfast * B1 * B1
    r_R2_R2p    = kfast * R2 * R2
    r_G2_G2p    = kfast * G2 * G2
    r_B2_B2p    = kfast * B2 * B2
    r_Y_Rp      = kfast * Yo * Yo
    r_X_Bp      = kfast * X  * X
 
    # "2R1' -> empty" (indicators self-degrade)
    r_R1p_deg   = kfast * R1p * R1p
    r_G1p_deg   = kfast * G1p * G1p
    r_B1p_deg   = kfast * B1p * B1p
    r_R2p_deg   = kfast * R2p * R2p
    r_G2p_deg   = kfast * G2p * G2p
    r_B2p_deg   = kfast * B2p * B2p
    r_Rp_deg    = kfast * Rp  * Rp
    r_Bp_deg    = kfast * Bp  * Bp
 
    # Group 4: absence indicators
    # "2Sr -> 2Sr + r"  (r is constantly generated)
    # "R' + r -> R'"    (r is consumed when red present)
    # Net: r only accumulates when R1p, R2p, Rp all = 0
    r_Sr_r      = kslow * Sr * Sr
    r_Sg_g      = kslow * Sg * Sg
    r_Sb_b      = kslow * Sb * Sb
 
    r_R1p_r     = kfast * R1p * r
    r_G1p_g     = kfast * G1p * g
    r_B1p_b     = kfast * B1p * b
    r_R2p_r     = kfast * R2p * r
    r_G2p_g     = kfast * G2p * g
    r_B2p_b     = kfast * B2p * b
    r_Rp_r      = kfast * Rp  * r
    r_Bp_b      = kfast * Bp  * b
 
    # Group 5: positive feedback kinetics
    # "R1' + X -> A"
    # Why: speeds up X->A transfer when red phase present
    r_R1p_X_A   = kfast * R1p * X
    r_G1p_R1_G1 = kfast * G1p * R1
    r_B1p_G1_B1 = kfast * B1p * G1
    r_R1p_B1_Y  = kfast * R1p * B1
 
    r_R2p_F_R2  = kfast * R2p * F
    r_G2p_R2_G2 = kfast * G2p * R2
    r_B2p_G2_B2 = kfast * B2p * G2
    r_R2p_B2_Y  = kfast * R2p * B2
 
    # ── ODEs: d[species]/dt ─────────────────────────
    # Each species accumulates from reactions that
    # PRODUCE it and drains from reactions that CONSUME it
 
    dX   = - r_gX_A - r_X_Bp - r_R1p_X_A
    # X is consumed when: transferred to A (gX->A),
    # generating Bp indicator, or pulled by R1p feedback
 
    dA   = (+ r_gX_A          # gained from X transfer
            + r_H_A            # gained from H feedback
            + r_F_A            # gained from F feedback
            + r_R1p_X_A        # gained from positive feedback
            - r_A_Y            # lost to Y (1/8 scaling)
            - r_A_R1)          # lost into first delay R1
 
    dR1  = (+ r_A_R1           # gained from A
            - r_R1_CF          # lost when outputting C and F
            - r_bR1_G1         # lost when transitioning to G1
            - r_R1_R1p         # lost generating indicator
            - r_G1p_R1_G1)     # lost via positive feedback
 
    dG1  = (+ r_bR1_G1         # gained from R1 phase transition
            + r_G1p_R1_G1      # gained via positive feedback
            - r_rG1_B1         # lost transitioning to B1
            - r_G1_G1p         # lost generating indicator
            - r_B1p_G1_B1)     # lost via positive feedback
 
    dB1  = (+ r_rG1_B1         # gained from G1 transition
            + r_B1p_G1_B1      # gained via positive feedback
            - r_gB1_Y          # lost contributing to Y
            - r_B1_B1p         # lost generating indicator
            - r_R1p_B1_Y)      # lost via positive feedback to Y
 
    dC   = (+ r_R1_CF          # gained as copy of R1 output
            - r_C_Y)           # lost contributing to Y (1/8)
 
    dF   = (+ r_R1_CF          # gained as copy of R1 output
            - r_F_R2           # lost feeding into R2
            - r_F_A            # lost in feedback to A
            - r_R2p_F_R2)      # lost via R2 positive feedback
 
    dR2  = (+ r_F_R2           # gained from F
            + r_R2p_F_R2       # gained via positive feedback
            - r_R2_EH          # lost outputting E and H
            - r_bR2_G2         # lost transitioning to G2
            - r_R2_R2p         # lost generating indicator
            - r_G2p_R2_G2)     # lost via positive feedback
 
    dG2  = (+ r_bR2_G2         # gained from R2 transition
            + r_G2p_R2_G2      # gained via positive feedback
            - r_rG2_B2         # lost transitioning to B2
            - r_G2_G2p         # lost generating indicator
            - r_B2p_G2_B2)     # lost via positive feedback
 
    dB2  = (+ r_rG2_B2         # gained from G2 transition
            + r_B2p_G2_B2      # gained via positive feedback
            - r_gB2_Y          # lost contributing to Y
            - r_B2_B2p         # lost generating indicator
            - r_R2p_B2_Y)      # lost via positive feedback to Y
 
    dE   = (+ r_R2_EH          # gained as copy of R2 output
            - r_E_Y)           # lost contributing to Y (1/8)
 
    dH   = (+ r_R2_EH          # gained as copy of R2 output
            - r_H_A)           # lost in feedback to A (1/8)
 
    dY   = (+ r_A_Y            # gained from A (1/8 * A)
            + r_C_Y            # gained from C (1/8 * C)
            + r_E_Y            # gained from E (1/8 * E)
            + r_gB1_Y          # gained from B1 phase end
            + r_gB2_Y          # gained from B2 phase end
            + r_R1p_B1_Y       # gained via positive feedback
            + r_R2p_B2_Y)      # gained via positive feedback
 
    # Indicators — track concentration of each color phase
    dR1p = + r_R1_R1p - r_R1p_deg - r_R1p_r
    dG1p = + r_G1_G1p - r_G1p_deg - r_G1p_g
    dB1p = + r_B1_B1p - r_B1p_deg - r_B1p_b
    dR2p = + r_R2_R2p - r_R2p_deg - r_R2p_r
    dG2p = + r_G2_G2p - r_G2p_deg - r_G2p_g
    dB2p = + r_B2_B2p - r_B2p_deg - r_B2p_b
    dRp  = + r_Y_Rp   - r_Rp_deg  - r_Rp_r
    dBp  = + r_X_Bp   - r_Bp_deg  - r_Bp_b
 
    # Absence indicators — only present when their
    # corresponding color phase concentration is zero
    dr   = (+ r_Sr_r
            - r_R1p_r - r_R2p_r - r_Rp_r)
    dg   = (+ r_Sg_g
            - r_G1p_g - r_G2p_g)
    db   = (+ r_Sb_b
            - r_B1p_b - r_B2p_b - r_Bp_b)
 
    return [dX, dA, dR1, dG1, dB1, dC, dF,
            dR2, dG2, dB2, dE, dH, dY,
            dR1p, dG1p, dB1p, dR2p, dG2p, dB2p,
            dRp, dBp, dr, dg, db]
 
 
def run_cycle(X_input, state, cycle_num):
    """
    Run one full RGB cycle of the biquad filter.
 
    Parameters:
        X_input  : the input signal concentration for this cycle
        state    : dict of all current species concentrations
                   (carries over delay state from previous cycle)
        cycle_num: which cycle we are on (for printing)
 
    Returns:
        Y_out    : the output Y concentration after this cycle
        new_state: updated species concentrations to carry forward
    """
 
    print(f"\n{'='*55}")
    print(f"  CYCLE {cycle_num}  |  Input X = {X_input}")
    print(f"{'='*55}")
 
    # Build initial concentration vector for this cycle
    # X is set to the new input value
    # Y is reset to 0 (external sink consumed it)
    # All delay states (R1,G1,B1,R2,G2,B2) carry over
    y0 = [
        X_input,        #  0: X   — new input
        state['A'],     #  1: A
        state['R1'],    #  2: R1  — delay 1 state carries over
        state['G1'],    #  3: G1
        state['B1'],    #  4: B1
        state['C'],     #  5: C
        state['F'],     #  6: F
        state['R2'],    #  7: R2  — delay 2 state carries over
        state['G2'],    #  8: G2
        state['B2'],    #  9: B2
        state['E'],     # 10: E
        state['H'],     # 11: H
        0.0,            # 12: Y   — reset to 0 each cycle
        state['R1p'],   # 13: R1'
        state['G1p'],   # 14: G1'
        state['B1p'],   # 15: B1'
        state['R2p'],   # 16: R2'
        state['G2p'],   # 17: G2'
        state['B2p'],   # 18: B2'
        state['Rp'],    # 19: R'
        state['Bp'],    # 20: B'
        state['r'],     # 21: r
        state['g'],     # 22: g
        state['b'],     # 23: b
    ]
 
    print(f"\n  Initial state going into this cycle:")
    print(f"    X={X_input:.2f}  A={state['A']:.2f}  "
          f"R1={state['R1']:.2f}  G1={state['G1']:.2f}  B1={state['B1']:.2f}")
    print(f"    R2={state['R2']:.2f}  G2={state['G2']:.2f}  "
          f"B2={state['B2']:.2f}  Y=0.00 (reset)")
 
    # Time points — run long enough for reactions to complete
    t = np.linspace(0, 20, 2000)
 
    # Solve the ODE system
    sol = odeint(biquad_odes, y0, t, mxstep=5000)
 
    # Read off final concentrations
    Y_out = sol[-1, 12]
 
    print(f"\n  Reaction progress (selected time points):")
    print(f"  {'Time':<8} {'X':<8} {'A':<8} {'R1':<8} "
          f"{'B1':<8} {'R2':<8} {'B2':<8} {'Y':<8}")
    print(f"  {'-'*64}")
    for i in [0, 200, 500, 1000, 1500, 1999]:
        print(f"  {t[i]:<8.2f} "
              f"{sol[i,0]:<8.2f} "
              f"{sol[i,1]:<8.2f} "
              f"{sol[i,2]:<8.2f} "
              f"{sol[i,4]:<8.2f} "
              f"{sol[i,7]:<8.2f} "
              f"{sol[i,9]:<8.2f} "
              f"{sol[i,12]:<8.2f}")
 
    print(f"\n  >>> OUTPUT Y = {Y_out:.4f}")
 
    # Build new state to carry forward to next cycle
    # Y is NOT carried forward (it gets reset)
    # Everything else carries over
    keys = ['X','A','R1','G1','B1','C','F',
            'R2','G2','B2','E','H','Y',
            'R1p','G1p','B1p','R2p','G2p','B2p',
            'Rp','Bp','r','g','b']
    new_state = {k: max(sol[-1, i], 0)
                 for i, k in enumerate(keys)}
    new_state['Y'] = 0.0   # reset Y for next cycle
 
    return Y_out, new_state
 
 
# ─────────────────────────────────────────────────────
# MAIN: run all 5 cycles
# ─────────────────────────────────────────────────────
 
print("\nBIQUAD FILTER CRN SIMULATION")
print("Filter equation:")
print("  A[n]  = X[n] + (1/8)*D1[n-1] + (1/8)*D2[n-1]")
print("  D1[n] = A[n]    (first delay)")
print("  D2[n] = D1[n-1] (second delay)")
print("  Y[n]  = (1/8)*A[n] + (1/8)*D1[n-1] + (1/8)*D2[n-1]")
 
# Initial state — all species at 0 except absence
# indicators which start active (no signal present yet)
initial_state = {
    'X':0,'A':0,'R1':0,'G1':0,'B1':0,
    'C':0,'F':0,'R2':0,'G2':0,'B2':0,
    'E':0,'H':0,'Y':0,
    'R1p':0,'G1p':0,'B1p':0,
    'R2p':0,'G2p':0,'B2p':0,
    'Rp':0,'Bp':0,
    'r':1.0,   # absence indicators start ON
    'g':1.0,   # (no signal present at start)
    'b':1.0
}
 
inputs  = [100, 5, 500, 20, 250]
outputs = []
state   = initial_state
 
for cycle, X_val in enumerate(inputs, start=1):
    Y_out, state = run_cycle(X_val, state, cycle)
    outputs.append(Y_out)
 
# ── Summary ──────────────────────────────────────────
print(f"\n{'='*55}")
print("  FINAL SUMMARY")
print(f"{'='*55}")
print(f"  {'Cycle':<8} {'Input X':<12} {'Output Y':<12}")
print(f"  {'-'*32}")
for i, (x, y) in enumerate(zip(inputs, outputs), 1):
    print(f"  {i:<8} {x:<12} {y:<12.4f}")
 
# ── Manual verification ──────────────────────────────
print(f"\n{'='*55}")
print("  MANUAL VERIFICATION (difference equation)")
print(f"{'='*55}")
D1, D2 = 0, 0
print(f"  {'Cycle':<8} {'X':<8} {'A':<10} {'D1':<10} "
      f"{'D2':<10} {'Y':<10}")
print(f"  {'-'*55}")
for i, X in enumerate(inputs, 1):
    A  = X + (1/8)*D1 + (1/8)*D2
    Y  = (1/8)*A + (1/8)*D1 + (1/8)*D2
    D1_new = A
    D2_new = D1
    print(f"  {i:<8} {X:<8} {A:<10.4f} {D1:<10.4f} "
          f"{D2:<10.4f} {Y:<10.4f}")
    D1, D2 = D1_new, D2_new
 
# ── Plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
cycles = list(range(1, 6))
ax.plot(cycles, inputs,  'o-', label='Input X',  color='steelblue')
ax.plot(cycles, outputs, 's-', label='Output Y (CRN)', color='coral')
ax.set_xlabel('Cycle')
ax.set_ylabel('Concentration')
ax.set_title('Biquad Filter CRN — Input vs Output')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('biquad_crn_plot.png', dpi=150)
print("\n  Plot saved to biquad_crn_plot.png")
