//! Complex whitening using the Hermitian covariance E[x x^H].
//!
//! Mirrors the Python `whiten_hermitian` (which mirrors GIFT's icatb_pca_whitening.m).
//!
//! The Strong Uncorrelating Transform is deliberately NOT ported: no estimator calls it
//! (both estimators whiten internally), so porting it would be dead code.

use nalgebra::DMatrix;
use num_complex::Complex64;

pub struct Whitening {
    pub xw: DMatrix<Complex64>,         // (N, T)
    pub w_whiten: DMatrix<Complex64>,   // (N, P)
    pub w_dewhiten: DMatrix<Complex64>, // (P, N)
}

pub fn whiten_hermitian(x: &DMatrix<Complex64>, n_components: usize) -> Result<Whitening, String> {
    let (p, t) = x.shape();
    if n_components > p {
        return Err(format!(
            "whiten_hermitian: n_components={n_components} exceeds the {p} available rows"
        ));
    }
    let n = n_components;

    // centre along the sample axis
    let mut xc = x.clone();
    for mut row in xc.row_iter_mut() {
        let mean: Complex64 = row.iter().sum::<Complex64>() / Complex64::new(t as f64, 0.0);
        row.iter_mut().for_each(|z| *z -= mean);
    }

    // Hermitian covariance (conjugate transpose - NOT the plain transpose)
    let r = (&xc * xc.adjoint()) / Complex64::new(t as f64, 0.0);

    let eig = r.clone().symmetric_eigen();
    // sort eigenvalues descending, keep the top N
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .expect("eigenvalues are real and finite")
    });
    let keep = &order[..n];

    let d_max = eig.eigenvalues[order[0]];
    let d_min = eig.eigenvalues[*keep.last().expect("n >= 1")];
    if !(d_min > 1e-12 * d_max) {
        return Err(format!(
            "whiten_hermitian: data is rank-deficient at n_components={n} (smallest \
             retained eigenvalue {d_min:.3e} vs largest {d_max:.3e}); refusing to return \
             a silently NaN-contaminated result"
        ));
    }

    let mut w_whiten = DMatrix::<Complex64>::zeros(n, p);
    let mut w_dewhiten = DMatrix::<Complex64>::zeros(p, n);
    for (k, &idx) in keep.iter().enumerate() {
        let s = eig.eigenvalues[idx].sqrt();
        for i in 0..p {
            let u = eig.eigenvectors[(i, idx)];
            w_whiten[(k, i)] = u.conj() / Complex64::new(s, 0.0);
            w_dewhiten[(i, k)] = u * Complex64::new(s, 0.0);
        }
    }

    let xw = &w_whiten * &xc;
    Ok(Whitening {
        xw,
        w_whiten,
        w_dewhiten,
    })
}
