# Vehicle-monitoring dataset

Put the US end-to-end subset of the [OpenALPR benchmark](https://github.com/openalpr/benchmarks) here.
It is not redistributed in this repository.

```bash
git clone https://github.com/openalpr/benchmarks.git openalpr_benchmarks
cp openalpr_benchmarks/endtoend/us/*.jpg openalpr_benchmarks/endtoend/us/*.txt vehicle_dataset/
```

Expected layout (222 image/label pairs):

```
vehicle_dataset/
├── us1.jpg
├── us1.txt        # us1.jpg <TAB> x <TAB> y <TAB> w <TAB> h <TAB> PLATE
├── ...
```

`STAR_Vehicle_Monitoring.ipynb` reads the plate string from the last tab-separated field of each `.txt`.
