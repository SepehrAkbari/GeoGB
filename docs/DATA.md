# Data and experiment artifacts

Dylan Peifer's thesis archive is available from Zenodo record 4724144. The
compressed archive is 17.2 GB and has MD5 checksum
`9cf4d05d5b71825f947befd4846feeb5`.

Keep the data outside Git and retain the original archive and checksum. An
external SSD is suitable. For example:

```text
/Volumes/Research/GeoGB-data/
├── archives/deepgroebner-data.tar.gz
├── upstream-v3/                     # unmodified extraction
└── experiments/                     # new GeoGB runs
```

Download with resume and checksum verification:

```bash
scripts/download_dylan_data.sh /Volumes/Research/GeoGB-data/archives
```

The original programs expect a repository-local `data/` path. After extracting
the archive and locating its `data` directory, create a local symbolic link:

```bash
ln -s /absolute/path/to/deepgroebner/data data
```

On the initial GeoGB machine, the source directory is
`/Volumes/Sepehr-SSD/deepgroebner-data/deepgroebner/data`.

The `data` path is ignored by Git. If the external drive is not mounted, the
link will be unavailable; mount it at the same path before running experiments.
Do not modify the `upstream-v3` copy. Write new checkpoints and logs under the
separate `experiments` directory or another versioned run directory.
