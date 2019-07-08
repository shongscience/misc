gcloud beta dataproc clusters create spark-mini \
--optional-components=ANACONDA,JUPYTER \
--metadata 'CONDA_PACKAGES="scipy numpy pandas pyarrow matplotlib seaborn",MINICONDA_VARIANT=2' \
--initialization-actions \
gs://dataproc-initialization-actions/python/conda-install.sh \
--enable-component-gateway \
--bucket shongdata \
--project pyspark-multiverse \
--region asia-northeast1 --zone asia-northeast1-a \
--master-machine-type custom-6-30720 --master-boot-disk-size 32GB \
--worker-machine-type n1-highmem-8 --worker-boot-disk-size 32GB --num-workers 2 \
--image-version 1.4 \
--properties spark:spark.jars.packages=graphframes:graphframes:0.7.0-spark2.4-s_2.11
