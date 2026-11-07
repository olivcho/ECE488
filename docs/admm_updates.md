# ADMM Update Equations

Each iteration runs the updates in this order: `u`, `x`, `w`, `v`, then the dual variables.

## Original problem

$$
\min_{v \ge 0} \; \tfrac{1}{2}\|b - CMv\|_2^2 + \tau\|\Psi v\|_1
$$

Introducing helper variables gives the equivalent problem:

$$
\min_{v,\,x,\,u,\,w \ge 0} \; \tfrac{1}{2}\|b - Cx\|_2^2 + \tau\|u\|_1
\quad \text{subject to} \quad x = Mv,\; u = \Psi v,\; w = v
$$

## Symbols

### Fixed for the whole run

| Symbol | Meaning |
|---|---|
| `b` | Measured lensless image |
| `h` | Point spread function (PSF) |
| `M` | Convolution with the PSF: `Mv = h * v` |
| `C` | Crop from the full plane to the sensor |
| `Cᴴ` | Zero-padding from the sensor to the full plane |
| `Ψ` | Image gradient: horizontal and vertical pixel differences |
| `τ` | Strength of the TV penalty. Changes the answer |
| `μ₁` | Penalty strength for `x = Mv` |
| `μ₂` | Penalty strength for `u = Ψ v` |
| `μ₃` | Penalty strength for `w = v` |

### Change every iteration

| Symbol | Meaning | Starts as |
|---|---|---|
| `v` | Image estimate | A simple guess |
| `x` | Full-plane lensless image that should equal `Mv` | Computed in the first iteration |
| `u` | Sparse gradients that should equal `Ψ v` | Computed in the first iteration |
| `w` | Nonnegative image that should equal `v` | Computed in the first iteration |
| `ξ` | Running disagreement for `x = Mv` | 0 |
| `η` | Running disagreement for `u = Ψ v` | 0 |
| `ρ` | Running disagreement for `w = v` | 0 |

`v`, `x`, `w`, `ξ` and `ρ` live on the padded full plane. `u` and `η` have one horizontal and one vertical channel. `b` has the sensor's size.

## Updates

Each helper minimizes its own requirement plus a penalty. The penalty has a spring term and a dual term.

### 1. Update `u`

**Uses:** `v, η, τ, μ₂`

$$
u \leftarrow \arg\min_u \; \tau\|u\|_1 + \frac{\mu_2}{2}\|\Psi v - u\|_2^2 + \eta^H(\Psi v - u)
$$

**Solution:**

$$
u \leftarrow \mathcal{T}_{\tau/\mu_2}\!\left(\Psi v + \frac{\eta}{\mu_2}\right)
$$

`T` is soft-thresholding, applied to each value separately:

$$
\mathcal{T}_t(z) = \operatorname{sign}(z)\max(|z| - t,\, 0)
$$

### 2. Update `x`

**Uses:** `v, b, ξ, μ₁`

$$
x \leftarrow \arg\min_x \; \frac{1}{2}\|b - Cx\|_2^2 + \frac{\mu_1}{2}\|Mv - x\|_2^2 + \xi^H(Mv - x)
$$

**Solution:**

$$
x \leftarrow (C^HC + \mu_1 I)^{-1}\left(\xi + \mu_1 Mv + C^Hb\right)
$$

`CᴴC` is diagonal, with 1 on sensor pixels and 0 elsewhere, so this is a per-pixel division:

- **On the sensor:** divide by `1 + μ₁`
- **Off the sensor:** divide by `μ₁`, where `Cᴴb = 0`

### 3. Update `w`

**Uses:** `v, ρ, μ₃`

$$
w \leftarrow \arg\min_{w \ge 0} \; \frac{\mu_3}{2}\|v - w\|_2^2 + \rho^H(v - w)
$$

**Solution:**

$$
w \leftarrow \max\!\left(v + \frac{\rho}{\mu_3},\, 0\right)
$$

### 4. Update `v`

**Uses:** `x, u, w, ξ, η, ρ, μ₁, μ₂, μ₃`

`v` has no requirement of its own. It only minimizes the three penalties, using the new `x`, `u` and `w`:

$$
v \leftarrow \arg\min_v \;
\frac{\mu_1}{2}\|Mv - x\|_2^2 + \xi^H(Mv - x)
+ \frac{\mu_2}{2}\|\Psi v - u\|_2^2 + \eta^H(\Psi v - u)
+ \frac{\mu_3}{2}\|v - w\|_2^2 + \rho^H(v - w)
$$

**Solution:**

$$
r = (\mu_3 w - \rho) + \Psi^H(\mu_2 u - \eta) + M^H(\mu_1 x - \xi)
$$

$$
v \leftarrow \left(\mu_1 M^HM + \mu_2 \Psi^H\Psi + \mu_3 I\right)^{-1} r
$$

`MᴴM` and `ΨᴴΨ` are diagonal in Fourier space when both use circular boundaries on the padded plane, so the inverse is an element-wise division:

$$
v \leftarrow \mathcal{F}^{-1}\!\left[
\frac{\mathcal{F}(r)}{\mu_1|H|^2 + \mu_2|\tilde{\Psi}|^2 + \mu_3}
\right]
$$

- `H = F(h)`
- `|Ψ̃|²` is the Fourier transform of the Laplacian kernel `ΨᴴΨ`

### 5. Update the dual variables

**Uses:** the new `v`, plus `x, u, w, ξ, η, ρ, μ₁, μ₂, μ₃`

$$
\xi \leftarrow \xi + \mu_1(Mv - x)
$$

$$
\eta \leftarrow \eta + \mu_2(\Psi v - u)
$$

$$
\rho \leftarrow \rho + \mu_3(v - w)
$$

## Output

After the last iteration, the reconstruction is `Cv`, the image cropped to the sensor size.

## Dependencies

| Update | Depends on |
|---|---|
| `u` | `v, η` |
| `x` | `v, ξ, b` |
| `w` | `v, ρ` |
| `v` | new `u, x, w` and `ξ, η, ρ` |
| `ξ` | new `v, x` and old `ξ` |
| `η` | new `v, u` and old `η` |
| `ρ` | new `v, w` and old `ρ` |

No values for `τ` or `μ₁, μ₂, μ₃` are given here, because they have not been tuned for this dataset.
