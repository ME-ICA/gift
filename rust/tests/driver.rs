mod common;

use common::{fixtures_dir, TestRng};
use complex_gift::complex_io::{read_complex, ComplexType};
use complex_gift::driver::{run_from_files, unmask};
use complex_gift::pipeline::Estimator;
use nalgebra::DMatrix;
use num_complex::Complex64;

#[test]
fn unmask_restores_full_volume_length_with_zeros_outside() {
    let dims = [4usize, 4, 2];
    let v = dims[0] * dims[1] * dims[2];
    let mask: Vec<bool> = (0..v).map(|i| i % 2 == 0).collect();
    let kept = mask.iter().filter(|&&m| m).count();

    let s =
        DMatrix::<Complex64>::from_fn(2, kept, |i, j| Complex64::new((i * kept + j) as f64, 1.0));
    let vols = unmask(&s, &mask, dims);

    assert_eq!(vols.len(), 2);
    for (i, vol) in vols.iter().enumerate() {
        assert_eq!(vol.len(), v);
        let mut c = 0usize;
        for j in 0..v {
            if mask[j] {
                assert!(
                    (vol[j] - s[(i, c)]).norm() < 1e-12,
                    "row {i} voxel {j}: got {:?}, want {:?}",
                    vol[j],
                    s[(i, c)]
                );
                c += 1;
            } else {
                assert_eq!(vol[j], Complex64::new(0.0, 0.0)); // zero, not garbage
            }
        }
        assert_eq!(c, kept, "not every masked voxel was filled");
    }
}

/// Masked voxels must be scattered back in ASCENDING flat-voxel order. Reversing that order
/// (or any other permutation) leaves lengths, zeros and energy identical, so only an
/// order-sensitive fixture catches it.
#[test]
fn unmask_preserves_voxel_order() {
    let dims = [2usize, 2, 2];
    let mask = vec![true, false, true, true, false, false, true, false];
    // distinct, monotone values so any permutation is visible
    let s = DMatrix::<Complex64>::from_row_slice(
        1,
        4,
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(20.0, 0.0),
            Complex64::new(30.0, 0.0),
            Complex64::new(40.0, 0.0),
        ],
    );
    let vols = unmask(&s, &mask, dims);
    let got: Vec<f64> = vols[0].iter().map(|z| z.re).collect();
    assert_eq!(got, vec![10.0, 0.0, 20.0, 30.0, 0.0, 0.0, 40.0, 0.0]);
}

#[test]
fn unmask_rejects_a_mask_that_does_not_match() {
    let dims = [2usize, 2, 2];

    // kept > ncols. Note this direction is caught by nalgebra's bounds check even without
    // the assert, so on its own it does not prove the guard exists.
    let s = DMatrix::<Complex64>::zeros(1, 3);
    let mask = vec![true; 8]; // 8 kept, but s has 3 columns
    let err = std::panic::catch_unwind(|| unmask(&s, &mask, dims));
    assert!(
        err.is_err(),
        "mismatched mask must not be accepted silently"
    );

    // kept < ncols is the SILENT direction: without the assert this returns a
    // plausible-looking volume having quietly dropped the trailing columns. This is what
    // actually pins the guard.
    let s = DMatrix::<Complex64>::zeros(1, 8);
    let mask: Vec<bool> = (0..8).map(|i| i < 3).collect(); // only 3 kept
    let err = std::panic::catch_unwind(|| unmask(&s, &mask, dims));
    assert!(
        err.is_err(),
        "unmask silently dropped map columns instead of rejecting the mask"
    );

    // and a mask whose length disagrees with dims outright
    let s = DMatrix::<Complex64>::zeros(1, 4);
    let mask = vec![true; 4]; // dims imply 8 voxels
    let err = std::panic::catch_unwind(|| unmask(&s, &mask, dims));
    assert!(err.is_err(), "mask length must match dims");
}

#[test]
fn run_from_files_rejects_no_subjects() {
    let err = run_from_files(
        &[],
        2,
        &std::env::temp_dir().join("cg_rust_driver_empty"),
        ComplexType::RealImag,
        Estimator::NcFastica,
        &fixtures_dir(),
    )
    .expect_err("no subjects must be rejected");
    assert!(err.contains("no subjects"), "unhelpful error: {err}");
}

#[test]
fn run_from_files_writes_component_maps() {
    let mut rng = TestRng::new(51);
    let dims = [10usize, 10, 4]; // 400 voxels
    let v = dims[0] * dims[1] * dims[2];
    let (n, v_sig, t, n_sub) = (3usize, 300usize, 60usize, 3usize);

    let dir = std::env::temp_dir().join("cg_rust_driver");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let smaps =
        DMatrix::<f64>::from_fn(n, v_sig, |_, _| rng.normal() * rng.normal().abs().powf(1.5));
    let scale = smaps.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    let smaps = smaps / scale;
    let phi: Vec<f64> = (0..v_sig)
        .map(|_| (rng.uniform() * 20.0 - 10.0) / 180.0 * std::f64::consts::PI)
        .collect();

    // write each subject as an R_/I_ pair holding a (dims, T) volume
    let mut pairs = Vec::new();
    for s in 0..n_sub {
        let tc = DMatrix::<f64>::from_fn(t, n, |_, _| rng.normal());
        let modulation = &tc * &smaps;
        let mut flat = vec![Complex64::new(0.0, 0.0); v * t];
        for tt in 0..t {
            for vv in 0..v {
                let z = if vv < v_sig {
                    // signal: positive magnitude x STATIC per-voxel phase -> phase-stable
                    let mag = 100.0 + 10.0 * rng.uniform() + 5.0 * modulation[(tt, vv)];
                    Complex64::new(mag, 0.0) * Complex64::new(phi[vv].cos(), phi[vv].sin())
                } else {
                    // noise: random phase per timepoint -> the mask must drop it
                    let mag = 0.5 + rng.uniform();
                    let ang = std::f64::consts::TAU * rng.uniform();
                    Complex64::new(mag * ang.cos(), mag * ang.sin())
                };
                flat[tt * v + vv] = z; // this buffer's own convention; write_4d maps it out
            }
        }
        let f1 = dir.join(format!("R_sub{s:02}.nii"));
        let f2 = dir.join(format!("I_sub{s:02}.nii"));
        write_4d(&flat, dims, t, &f1, &f2);
        pairs.push((f1, f2));
    }

    let out = dir.join("out");
    let (res, written) = run_from_files(
        &pairs,
        n,
        &out,
        ComplexType::RealImag,
        Estimator::Cebm { seed: 0 },
        &fixtures_dir(),
    )
    .expect("driver");

    // The driver must reconstruct each subject's (T, V) matrix from the file's flat buffer.
    // A wrong flattening (e.g. time-slowest `data[tt*v + vv]`) is SILENT: it still yields a
    // (T, V) matrix of the right shape and the pipeline still runs. It shows up here,
    // because scrambling destroys the per-voxel phase stability the mask keys on.
    let kept_sig = res.mask[..v_sig].iter().filter(|&&m| m).count() as f64 / v_sig as f64;
    let kept_noise = res.mask[v_sig..].iter().filter(|&&m| m).count() as f64 / (v - v_sig) as f64;
    assert!(kept_sig > 0.9, "mask dropped signal voxels: {kept_sig:.3}");
    assert!(kept_noise < 0.05, "mask kept noise voxels: {kept_noise:.3}");

    assert_eq!(written.len(), n);

    // Pin the FILE CONTENTS to the in-memory result. Without this the test passes when the
    // driver writes the wrong component to every file, writes conj/scaled maps, or permutes
    // voxels within the volume - existence + "some voxel is nonzero" cannot see any of that.
    //
    // Build `expected` HERE rather than calling `unmask`: deriving it from the same function
    // the driver uses makes any unmask bug cancel out on both sides and the assertion
    // vacuous. (Verified: with unmask's scatter order reversed, an `unmask`-derived expected
    // left this test green.) `unmask` itself is pinned by the tests above.
    let kept: Vec<usize> = (0..v).filter(|&j| res.mask[j]).collect();
    assert_eq!(kept.len(), res.s_group.ncols(), "mask and s_group disagree");
    let expected: Vec<Vec<Complex64>> = (0..n)
        .map(|i| {
            let mut vol = vec![Complex64::new(0.0, 0.0); v];
            for (c, &j) in kept.iter().enumerate() {
                vol[j] = res.s_group[(i, c)];
            }
            vol
        })
        .collect();

    for (k, (f1, f2)) in written.iter().enumerate() {
        assert!(f1.exists() && f2.exists());
        assert_eq!(
            f1.file_name().unwrap().to_str().unwrap(),
            format!("R_component_{:03}.nii", k + 1)
        );
        let (vals, d) = read_complex(f1, f2, ComplexType::RealImag).expect("read back");
        assert_eq!([d[0], d[1], d[2]], dims);
        assert_eq!(vals.len(), v);

        for j in 0..v {
            assert!(
                (vals[j] - expected[k][j]).norm() < 1e-9,
                "component {k} voxel {j}: file has {:?}, s_group implies {:?}",
                vals[j],
                expected[k][j]
            );
            if !res.mask[j] {
                assert_eq!(vals[j], Complex64::new(0.0, 0.0));
            }
        }
        let energy: f64 = vals.iter().map(|z| z.norm_sqr()).sum();
        assert!(energy > 0.0, "component {k}: empty volume");
    }

    // the components must be DISTINCT volumes - writing vols[0] n times would otherwise
    // satisfy every per-file assertion above if s_group's rows happened to be compared
    // against themselves.
    for a in 0..n {
        for b in (a + 1)..n {
            let d: f64 = expected[a]
                .iter()
                .zip(expected[b].iter())
                .map(|(x, y)| (x - y).norm_sqr())
                .sum();
            assert!(d > 0.0, "components {a} and {b} are identical");
        }
    }
}

/// Write a (V*T) buffer indexed `flat[tt * v + vv]` as a 4-D R_/I_ NIfTI pair.
fn write_4d(
    flat: &[Complex64],
    dims: [usize; 3],
    t: usize,
    first: &std::path::Path,
    second: &std::path::Path,
) {
    use nifti::writer::WriterOptions;
    let v = dims[0] * dims[1] * dims[2];
    let mut re = ndarray::Array4::<f64>::zeros((dims[0], dims[1], dims[2], t));
    let mut im = ndarray::Array4::<f64>::zeros((dims[0], dims[1], dims[2], t));
    for tt in 0..t {
        for vv in 0..v {
            // The crate's flat voxel index is C-order (x SLOWEST) - read_volume calls
            // .as_standard_layout() precisely to guarantee this, and write_complex builds
            // its Array3 from a C-order buffer. Mapping vv x-fastest here (x = vv % nx)
            // would silently disagree with the driver that reads these files back.
            let z = vv % dims[2];
            let y = (vv / dims[2]) % dims[1];
            let x = vv / (dims[1] * dims[2]);
            re[(x, y, z, tt)] = flat[tt * v + vv].re;
            im[(x, y, z, tt)] = flat[tt * v + vv].im;
        }
    }
    WriterOptions::new(first).write_nifti(&re).unwrap();
    WriterOptions::new(second).write_nifti(&im).unwrap();
}

/// The maps must inherit the INPUT's geometry. Without this they are numerically correct
/// but carry an identity affine, so they do not overlay on the subject's anatomy - and
/// nothing else in the suite can see it, because every other assertion is array-level.
/// Python's `write_complex(..., affine=affine)` propagates it, so Rust must too or the two
/// ports are not interchangeable on real data.
#[test]
fn written_maps_inherit_the_input_geometry() {
    use nifti::{NiftiHeader, NiftiObject, ReaderOptions};

    let dir = std::env::temp_dir().join("cg_rust_driver_affine");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let dims = [4usize, 3, 2];
    let v = dims[0] * dims[1] * dims[2];
    let (n, t, n_sub) = (2usize, 24usize, 2usize);
    let mut rng = TestRng::new(9);

    // a deliberately NON-identity geometry: 2mm/3mm/4mm voxels, translated origin
    let href = NiftiHeader {
        pixdim: [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        sform_code: 1,
        srow_x: [2.0, 0.0, 0.0, -10.0],
        srow_y: [0.0, 3.0, 0.0, -20.0],
        srow_z: [0.0, 0.0, 4.0, -30.0],
        ..Default::default()
    };

    // phase-stable signal everywhere, so the mask keeps voxels and the pipeline runs
    let phi: Vec<f64> = (0..v)
        .map(|_| (rng.uniform() * 20.0 - 10.0) / 180.0 * std::f64::consts::PI)
        .collect();
    let mut pairs = Vec::new();
    for s in 0..n_sub {
        let mut re = ndarray::Array4::<f64>::zeros((dims[0], dims[1], dims[2], t));
        let mut im = ndarray::Array4::<f64>::zeros((dims[0], dims[1], dims[2], t));
        for tt in 0..t {
            for (vv, &ph) in phi.iter().enumerate() {
                let mag = 100.0
                    + 10.0 * rng.uniform()
                    + 5.0 * (rng.normal() * rng.normal().abs().powf(1.5));
                let z = Complex64::new(mag, 0.0) * Complex64::new(ph.cos(), ph.sin());
                let (x, y, zc) = (
                    vv / (dims[1] * dims[2]),
                    (vv / dims[2]) % dims[1],
                    vv % dims[2],
                );
                re[(x, y, zc, tt)] = z.re;
                im[(x, y, zc, tt)] = z.im;
            }
        }
        let f1 = dir.join(format!("R_s{s:02}.nii"));
        let f2 = dir.join(format!("I_s{s:02}.nii"));
        nifti::writer::WriterOptions::new(&f1)
            .reference_header(&href)
            .write_nifti(&re)
            .unwrap();
        nifti::writer::WriterOptions::new(&f2)
            .reference_header(&href)
            .write_nifti(&im)
            .unwrap();
        pairs.push((f1, f2));
    }

    let out = dir.join("out");
    let (_res, written) = run_from_files(
        &pairs,
        n,
        &out,
        ComplexType::RealImag,
        Estimator::NcFastica,
        &fixtures_dir(),
    )
    .expect("driver");

    for (f1, f2) in &written {
        for p in [f1, f2] {
            let obj = ReaderOptions::new().read_file(p).expect("read map");
            let h = obj.header();
            assert_eq!(
                h.srow_x,
                href.srow_x,
                "{}: srow_x not inherited",
                p.display()
            );
            assert_eq!(
                h.srow_y,
                href.srow_y,
                "{}: srow_y not inherited",
                p.display()
            );
            assert_eq!(
                h.srow_z,
                href.srow_z,
                "{}: srow_z not inherited",
                p.display()
            );
            assert_eq!(
                h.sform_code,
                href.sform_code,
                "{}: sform_code lost",
                p.display()
            );
            assert_eq!(
                &h.pixdim[1..4],
                &href.pixdim[1..4],
                "{}: voxel sizes lost",
                p.display()
            );
            // the maps are 3-D even though the inputs were 4-D
            assert_eq!(h.dim[0], 3, "{}: expected a 3-D map", p.display());
            assert_eq!(
                &h.dim[1..4],
                &[dims[0] as u16, dims[1] as u16, dims[2] as u16]
            );
        }
    }
}
