> Make sure you have docker installed.

Create persistent volume for the database

```bash
docker volume create mongo
```

Run the container, mounting the volume and forwarding the port.

```bash
docker run --name empirical-mongo -p 27017:27017 -v mongo:/data/db -d mongo
```

Install mongodb tools [here](https://www.mongodb.com/docs/database-tools/installation/installation/) and download the [dataset](https://zenodo.org/record/7182101).
Got to the `DataDump` folder and run
```bash
mongorestore --gzip --archive=mongodump-JiraRepos.archive --nsFrom "JiraRepos.*" --nsTo "JiraRepos.*"
```

If the command fails some time in you can run it again and it will skip all records it already inserted.