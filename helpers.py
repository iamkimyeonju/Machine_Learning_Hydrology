import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_network(input_dim, hidden_dims, feature_cols):
    layer_sizes = [input_dim] + list(hidden_dims) + [1]
    n_layers    = len(layer_sizes)
    MAX_SHOW    = 20
    R           = 0.28
    Y_LO, Y_HI = 1.0, 9.0

    COLORS = ['#4e79a7'] + ['#59a14f'] * len(hidden_dims) + ['#e15759']
    xs     = np.linspace(2.0, n_layers * 2.8, n_layers)

    def layer_ys(n):
        n_draw = min(n, MAX_SHOW)
        return np.array([(Y_LO + Y_HI) / 2]) if n_draw == 1 else np.linspace(Y_LO, Y_HI, n_draw)

    all_ys = [layer_ys(n) for n in layer_sizes]
    fig, ax = plt.subplots(figsize=(max(11, 2.8 * n_layers), 8))

    for l in range(n_layers - 1):
        for y1 in all_ys[l]:
            for y2 in all_ys[l + 1]:
                ax.plot([xs[l], xs[l+1]], [y1, y2],
                        color='#999999', lw=0.6, alpha=0.35, zorder=1)

    for l in range(n_layers):
        n, c = layer_sizes[l], COLORS[l]
        for y in all_ys[l]:
            ax.add_patch(plt.Circle((xs[l], y), R, facecolor=c, lw=1.5, zorder=3))

        if n > MAX_SHOW:
            ax.text(xs[l], (Y_LO + Y_HI) / 2, '⋮', ha='center', va='center',
                    fontsize=22, color=c, fontweight='bold', zorder=5)
            ax.text(xs[l], Y_HI + 0.5, f'({n} total)', ha='center', fontsize=8, color='#888888')

        if l == 0:
            for i, y in enumerate(all_ys[0]):
                ax.text(xs[0] - R - 0.12, y, feature_cols[i][:16],
                        ha='right', va='center', fontsize=7, color='#444444')

        if l == n_layers - 1:
            ax.text(xs[-1] + R + 0.12, all_ys[-1][0], 'Q90 [mm/day]',
                    ha='left', va='center', fontsize=9, color='#e15759', fontweight='bold')

        if l == 0:
            lname = f'Input\n({n} features)'
        elif l == n_layers - 1:
            lname = f'Output\n({n} unit)'
        else:
            lname = f'Hidden {l}\n({n} units)'
        ax.text(xs[l], Y_LO - 1.0, lname, ha='center', va='top',
                fontsize=9.5, fontweight='bold', color=c, multialignment='center')

        if 0 < l < n_layers - 1:
            ax.text(xs[l], Y_HI + 1.2, 'Linear → ReLU', ha='center',
                    fontsize=7.5, color='#666666', style='italic')
        elif l == n_layers - 1:
            ax.text(xs[l], Y_HI + 1.2, 'Linear (no activation)', ha='center',
                    fontsize=7.5, color='#666666', style='italic')

        if l > 0:
            n_par = layer_sizes[l-1] * n + n
            ax.text(xs[l], Y_HI + 2.0, f'{n_par:,} params', ha='center',
                    fontsize=8, color='#aaaaaa')

    total = sum(layer_sizes[l-1] * layer_sizes[l] + layer_sizes[l] for l in range(1, n_layers))
    arch  = ' → '.join(str(s) for s in layer_sizes)
    ax.set_xlim(min(xs) - 2.2, max(xs) + 2.2)
    ax.set_ylim(Y_LO - 2.0, Y_HI + 3.0)
    ax.axis('off')
    ax.set_title(f'MLP  {arch}  |  {total:,} parameters', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.show()
    print(f'Architecture : {arch}')
    print(f'Total parameters : {total:,}')


def draw_lstm_arch(input_size=6, hidden_size=64, num_layers=2, seq_len=365):
    """
    Vertical block diagram for the single-catchment LSTM (notebook 02).
    Shows tensor shapes on arrows between blocks.
    """
    BLUE   = '#4e79a7'
    GREEN  = '#59a14f'
    RED    = '#e15759'
    ORANGE = '#f28e2b'

    BW  = 7.8   # block width
    BH  = 1.25  # block height
    CX  = 5.0   # center x
    GAP = 0.75  # vertical gap between blocks

    # Build block list top-to-bottom: (title, subtitle, color)
    blocks = [
        ('Input Sequence',
         f'(batch, {seq_len}, {input_size})   ·   {input_size} forcing variables × {seq_len} days',
         BLUE),
    ]
    for l in range(1, num_layers + 1):
        in_sz  = input_size if l == 1 else hidden_size
        params = 4 * (in_sz + hidden_size) * hidden_size + 4 * hidden_size
        blocks.append((
            f'LSTM  Layer {l}',
            f'hidden_size = {hidden_size}   ·   {params:,} params   ·   forget / input / output gates',
            GREEN,
        ))
    blocks.append(('Linear',   f'{hidden_size} → 1   (no activation)',     RED))
    blocks.append(('Output  Q(t)', '(batch, 1)   predicted discharge',      ORANGE))

    n_blocks  = len(blocks)
    fig_h     = 2.5 + n_blocks * (BH + GAP)
    fig, ax   = plt.subplots(figsize=(9, fig_h))
    ax.axis('off')

    # y-positions (top block highest)
    total_h = n_blocks * BH + (n_blocks - 1) * GAP
    ys = [total_h + 0.6 - i * (BH + GAP) for i in range(n_blocks)]

    # Arrow labels between blocks
    arrow_labels = []
    for i in range(n_blocks - 1):
        if i == 0:
            lbl = f'(batch, {seq_len}, {input_size})'
        elif i == n_blocks - 2:
            lbl = f'(batch, {hidden_size})'
        else:
            lbl = f'(batch, {seq_len}, {hidden_size})'   # between LSTM layers
        arrow_labels.append(lbl)

    # Draw arrows first (under blocks)
    for i, lbl in enumerate(arrow_labels):
        y_from = ys[i]   - BH / 2
        y_to   = ys[i+1] + BH / 2
        ax.annotate('', xy=(CX, y_to + 0.06), xytext=(CX, y_from - 0.06),
                    arrowprops=dict(arrowstyle='->', color='#777777', lw=1.8))
        ax.text(CX + 0.28, (y_from + y_to) / 2, lbl,
                fontsize=8, color='#888888', va='center')

    # Draw blocks
    for (title, subtitle, color), yc in zip(blocks, ys):
        box = FancyBboxPatch((CX - BW/2, yc - BH/2), BW, BH,
                             boxstyle='round,pad=0.12',
                             facecolor=color, edgecolor='white', lw=2.5,
                             alpha=0.92, zorder=3)
        ax.add_patch(box)
        ax.text(CX, yc + 0.2,  title,    ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', zorder=4)
        ax.text(CX, yc - 0.22, subtitle, ha='center', va='center',
                fontsize=8, color='white', alpha=0.88, zorder=4)

    # Total params
    n_lstm  = sum(
        4 * ((input_size if l == 0 else hidden_size) + hidden_size) * hidden_size + 4 * hidden_size
        for l in range(num_layers)
    )
    n_total = n_lstm + hidden_size + 1

    ax.set_xlim(0, 10)
    ax.set_ylim(min(ys) - BH, max(ys) + BH)
    ax.set_title(
        f'LSTM Architecture  ·  {num_layers} layer{"s" if num_layers > 1 else ""}  ·  '
        f'hidden = {hidden_size}  ·  {n_total:,} total parameters',
        fontsize=12, fontweight='bold', pad=14,
    )
    plt.tight_layout()
    plt.show()
    print(f'Total parameters: {n_total:,}')


def draw_lstm_neurons(input_size=6, hidden_size=64, num_layers=2, feature_cols=None):
    """
    Neuron-and-connection diagram for the single-catchment LSTM (notebook 02).
    Shows input circles, gate circles (f/i/g/o per layer), hidden-state circles,
    and output circle, connected by weighted lines — same visual language as
    draw_network for the MLP.
    """
    MAX_SHOW_H = 10
    R          = 0.22
    Y_LO, Y_HI = 1.0, 9.0
    Y_MID       = (Y_LO + Y_HI) / 2

    INPUT_C  = '#4e79a7'
    GATE_CS  = ['#e15759', '#59a14f', '#f28e2b', '#b07aa1']   # f, i, g, o
    HIDDEN_C = '#76b7b2'
    FC_C     = '#e15759'

    gate_syms   = ['fₜ', 'iₜ', 'gₜ', 'oₜ']
    gate_titles = ['forget', 'input', 'cell', 'output']

    # ── X positions ───────────────────────────────────────────────────────────
    xs = {'input': 1.5}
    for l in range(1, num_layers + 1):
        x_prev = xs['input'] if l == 1 else xs[f'h{l-1}']
        xs[f'g{l}'] = x_prev + 3.2
        xs[f'h{l}'] = xs[f'g{l}'] + 2.5
    xs['out'] = xs[f'h{num_layers}'] + 2.8

    # ── Y positions ───────────────────────────────────────────────────────────
    def node_ys(n, max_show=None):
        nd = min(n, max_show) if max_show else n
        return np.linspace(Y_LO, Y_HI, nd) if nd > 1 else np.array([Y_MID])

    input_ys  = node_ys(input_size)
    gate_ys   = np.linspace(Y_LO + 1.2, Y_HI - 1.2, 4)
    hidden_ys = node_ys(hidden_size, MAX_SHOW_H)
    output_ys = np.array([Y_MID])

    # ── Helpers ───────────────────────────────────────────────────────────────
    def connections(ax, x1, ys1, x2, ys2, alpha=0.18, lw=0.5, color='#999999'):
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, lw=lw, zorder=1)

    def nodes(ax, x, ys, color, labels=None):
        for i, y in enumerate(ys):
            ax.add_patch(plt.Circle((x, y), R, facecolor=color,
                                     edgecolor='white', lw=1.5, zorder=3))
            if labels and i < len(labels):
                ax.text(x, y, labels[i], ha='center', va='center',
                        fontsize=7.5, color='white', fontweight='bold', zorder=4)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(xs['out'] + 2.2, 9))
    ax.axis('off')

    # ── Input column ──────────────────────────────────────────────────────────
    xi = xs['input']
    connections(ax, xi, input_ys, xs['g1'], gate_ys)
    nodes(ax, xi, input_ys, INPUT_C)
    if feature_cols:
        for y, lbl in zip(input_ys, feature_cols):
            ax.text(xi - R - 0.12, y, lbl[:12], ha='right', va='center',
                    fontsize=8, color='#444444')
    ax.text(xi, Y_LO - 0.9, 'Input  xₜ\n(6 vars)', ha='center', va='top',
            fontsize=9, fontweight='bold', color=INPUT_C, multialignment='center')

    # ── LSTM layers ───────────────────────────────────────────────────────────
    for l in range(1, num_layers + 1):
        xg = xs[f'g{l}']
        xh = xs[f'h{l}']

        # Gate → hidden connections
        connections(ax, xg, gate_ys, xh, hidden_ys)

        # Gate nodes — one circle per gate, each a different colour
        for gi, (sym, title, gc) in enumerate(zip(gate_syms, gate_titles, GATE_CS)):
            ax.add_patch(plt.Circle((xg, gate_ys[gi]), R + 0.05, facecolor=gc,
                                     edgecolor='white', lw=1.5, zorder=3))
            ax.text(xg, gate_ys[gi], sym, ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold', zorder=4)
            ax.text(xg + R + 0.18, gate_ys[gi], title, ha='left', va='center',
                    fontsize=7, color=gc)

        # Hidden-state nodes
        nodes(ax, xh, hidden_ys, HIDDEN_C)
        if hidden_size > MAX_SHOW_H:
            ax.text(xh, Y_MID, '⋮', ha='center', va='center',
                    fontsize=22, color=HIDDEN_C, fontweight='bold', zorder=5)
            ax.text(xh, Y_HI + 0.55, f'({hidden_size} units total)',
                    ha='center', fontsize=8, color='#888888')

        # Recurrent arrow: hₜ → gates of same layer (curved above)
        ax.annotate('', xy=(xg, gate_ys[-1] + 0.38), xytext=(xh, hidden_ys[-1] + 0.38),
                    arrowprops=dict(arrowstyle='->', color='#cc6677', lw=1.6,
                                    connectionstyle='arc3,rad=-0.35'))
        ax.text((xg + xh) / 2, gate_ys[-1] + 1.05,
                'hₜ₋₁  (recurrent)', ha='center', fontsize=8,
                color='#cc6677', style='italic')

        # Layer label + param count
        in_sz  = input_size if l == 1 else hidden_size
        params = 4 * (in_sz + hidden_size) * hidden_size + 8 * hidden_size
        ax.text((xg + xh) / 2, Y_LO - 0.9,
                f'LSTM Layer {l}\n{params:,} params',
                ha='center', va='top', fontsize=9, fontweight='bold',
                color=HIDDEN_C, multialignment='center')

        # Forward connection to next layer
        if l < num_layers:
            connections(ax, xh, hidden_ys, xs[f'g{l+1}'], gate_ys)

    # ── Output column ─────────────────────────────────────────────────────────
    xlast = xs[f'h{num_layers}']
    xo    = xs['out']
    connections(ax, xlast, hidden_ys, xo, output_ys)
    ax.text((xlast + xo) / 2, Y_MID + 0.6,
            'take hₜ at\nfinal step T', ha='center', fontsize=8,
            color='#888888', style='italic')
    nodes(ax, xo, output_ys, FC_C)
    ax.text(xo + R + 0.12, Y_MID, 'Q(t)', ha='left', va='center',
            fontsize=10, color=FC_C, fontweight='bold')
    ax.text(xo, Y_LO - 0.9, 'FC output\n(1 unit)', ha='center', va='top',
            fontsize=9, fontweight='bold', color=FC_C, multialignment='center')

    # ── Title ─────────────────────────────────────────────────────────────────
    n_lstm = sum(
        4 * ((input_size if l == 0 else hidden_size) + hidden_size) * hidden_size
        + 8 * hidden_size
        for l in range(num_layers)
    )
    n_total = n_lstm + hidden_size + 1
    ax.set_xlim(0, xo + 2.5)
    ax.set_ylim(Y_LO - 2.2, Y_HI + 2.5)
    ax.set_title(
        f'LSTM  ·  {num_layers} layers  ·  hidden = {hidden_size}  '
        f'·  {n_total:,} total parameters',
        fontsize=12, fontweight='bold', pad=10,
    )
    plt.tight_layout()
    plt.show()


def draw_ea_lstm_arch(dynamic_size=6, static_size=28, hidden_size=64, seq_len=365):
    """
    Architecture diagram for the full EA-LSTM (Kratzert et al. 2019, notebook 03).

    Left branch : static attrs → V_i (Linear, no bias) → s_proj
                  s_proj is added to the input gate at EVERY timestep
    Right branch: forcing sequence → into the EA-LSTM cell each step
    Bottom      : EA-LSTM → Linear → Q(t)

    The unrolled LSTM loop is shown as a recurrent block annotated with
    which gate receives the static signal.
    """
    BLUE   = '#4e79a7'
    GREEN  = '#59a14f'
    PURPLE = '#8e6bbf'
    RED    = '#e15759'
    ORANGE = '#f28e2b'
    GREY   = '#aaaaaa'

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.5, 11.2)
    ax.axis('off')

    def block(cx, cy, w, h, title, subtitle, color, alpha=0.92):
        box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                             boxstyle='round,pad=0.12',
                             facecolor=color, edgecolor='white', lw=2.5,
                             alpha=alpha, zorder=3)
        ax.add_patch(box)
        ax.text(cx, cy + 0.2,  title,    ha='center', va='center',
                fontsize=10.5, fontweight='bold', color='white', zorder=4)
        if subtitle:
            ax.text(cx, cy - 0.22, subtitle, ha='center', va='center',
                    fontsize=8, color='white', alpha=0.88, zorder=4)

    def varrow(x, y1, y2, label='', label_side=1):
        ax.annotate('', xy=(x, y2 + 0.06), xytext=(x, y1 - 0.06),
                    arrowprops=dict(arrowstyle='->', color='#777777', lw=1.8))
        if label:
            ax.text(x + 0.22 * label_side, (y1 + y2) / 2, label,
                    fontsize=8, color='#888888', va='center',
                    ha='left' if label_side > 0 else 'right')

    def carrow(x1, y1, x2, y2, label='', cs='arc3,rad=-0.2',
               lx_off=0.0, ly_off=0.3, dashed=False):
        ls = 'dashed' if dashed else 'solid'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#777777', lw=1.8,
                                    linestyle=ls,
                                    connectionstyle=cs))
        if label:
            mx = (x1 + x2) / 2 + lx_off
            my = (y1 + y2) / 2 + ly_off
            ax.text(mx, my, label, fontsize=8, color='#555555', va='center',
                    ha='center', bbox=dict(fc='white', ec='none', pad=1.5))

    # ── Layout ────────────────────────────────────────────────────────────────
    LX    = 3.0    # static branch x
    RX    = 11.0   # forcing branch x
    MX    = 7.0    # LSTM / output x
    BW_SM = 4.6
    BW_LG = 8.8
    BW_MD = 6.5
    BH    = 1.2

    # ── Left branch: static → V_i projection ─────────────────────────────────
    block(LX, 9.8, BW_SM, BH,
          'Static Attributes',
          f'(batch, {static_size})   ·   {static_size} catchment attributes',
          BLUE)
    varrow(LX, 9.2, 8.3)
    block(LX, 7.7, BW_SM, BH,
          'V_i  (Linear, no bias)',
          f'Linear({static_size} → {hidden_size})   —   no bias, no activation',
          PURPLE)
    # Label below encoder
    ax.text(LX, 6.95, f's_proj = V_i · s   shape: (batch, {hidden_size})',
            ha='center', fontsize=8, color='#666666', style='italic')

    # ── Right branch: forcing ──────────────────────────────────────────────────
    block(RX, 9.8, BW_SM, BH,
          'Forcing Sequence',
          f'(batch, {seq_len}, {dynamic_size})   ·   {dynamic_size} vars × {seq_len} days',
          BLUE)

    # ── EA-LSTM cell (unrolled loop hint) ─────────────────────────────────────
    LSTM_Y = 5.2
    n_cell = (4 * (dynamic_size + hidden_size) * hidden_size   # W_f, W_i, W_g, W_o
              + 4 * hidden_size                                 # biases
              + static_size * hidden_size)                      # V_i

    block(MX, LSTM_Y, BW_LG, 1.5,
          'EA-LSTM Cell  ×  365 steps',
          (f'f_t = σ(W_f[h,x]+b)   '
           f'i_t = σ(W_i[h,x]+b + s_proj)   '   # ← the key line
           f'g_t = tanh(W_g[h,x]+b)   '
           f'o_t = σ(W_o[h,x]+b)'),
          GREEN)

    # Annotation: which gate gets static
    ax.text(MX, LSTM_Y - 0.05,
            '↑ s_proj added here at every t',
            ha='center', fontsize=7.5, color='#ffdd88',
            fontweight='bold', zorder=5)

    # ── Converging arrows ─────────────────────────────────────────────────────
    # Static → LSTM (dashed, labelled)
    carrow(LX, 6.85, MX - 4.0, LSTM_Y + 0.55,
           label='s_proj  (batch, 64)\nadded to input gate\nat every timestep',
           cs='arc3,rad=-0.18', lx_off=-1.6, ly_off=0.5, dashed=True)

    # Forcing → LSTM
    carrow(RX, 9.2, MX + 4.0, LSTM_Y + 0.55,
           label=f'x_t  one step at a time\n(batch, {dynamic_size})',
           cs='arc3,rad=0.18', lx_off=1.6, ly_off=0.5)

    # Recurrent self-loop hint
    ax.annotate('', xy=(MX + 4.8, LSTM_Y + 0.3), xytext=(MX + 4.8, LSTM_Y - 0.3),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=1.2,
                                connectionstyle='arc3,rad=-0.6'))
    ax.text(MX + 5.4, LSTM_Y, 'h, c\nrecurrent', fontsize=7.5,
            color=GREY, ha='left', va='center', style='italic')

    # h₀ = c₀ = 0
    ax.text(MX, LSTM_Y + 1.0, 'h₀ = c₀ = 0', fontsize=8, color='#bbbbbb',
            ha='center', style='italic')

    # ── Linear & Output ───────────────────────────────────────────────────────
    varrow(MX, LSTM_Y - 0.75, LSTM_Y - 1.5, label=f'h_T  (batch, {hidden_size})')
    block(MX, LSTM_Y - 2.1, BW_MD, BH,
          'Linear',
          f'{hidden_size} → 1   (no activation)',
          RED)
    varrow(MX, LSTM_Y - 2.7, LSTM_Y - 3.4)
    block(MX, LSTM_Y - 4.0, 5.0, 1.0,
          'Output  Q(t)',
          '(batch, 1)   predicted discharge',
          ORANGE)

    # ── Section labels & dividers ─────────────────────────────────────────────
    ax.text(LX,  10.9, 'Static branch\n(computed once)', ha='center',
            fontsize=9, color='#888888', style='italic', multialignment='center')
    ax.text(RX,  10.9, 'Dynamic branch\n(one step at a time)', ha='center',
            fontsize=9, color='#888888', style='italic', multialignment='center')
    ax.text(MX,  10.9, 'Shared model', ha='center', fontsize=9,
            color='#888888', style='italic')

    for x in [5.3, 8.7]:
        ax.axvline(x, color='#e8e8e8', lw=1, ls='--', zorder=0)

    # ── Parameter summary ─────────────────────────────────────────────────────
    n_Vi     = static_size * hidden_size
    n_gates  = 4 * (dynamic_size + hidden_size) * hidden_size + 4 * hidden_size
    n_fc     = hidden_size + 1
    n_total  = n_Vi + n_gates + n_fc

    ax.set_title(
        f'EA-LSTM (Kratzert et al. 2019)  ·  static = {static_size}  ·  '
        f'dynamic = {dynamic_size}  ·  hidden = {hidden_size}  ·  {n_total:,} parameters',
        fontsize=12, fontweight='bold', pad=14,
    )
    plt.tight_layout()
    plt.show()
    print(f'V_i  (static → input gate) : {n_Vi:,} params')
    print(f'LSTM gates (W_f,W_i,W_g,W_o): {n_gates:,} params')
    print(f'Output head                : {n_fc:,} params')
    print(f'Total                      : {n_total:,} params')
