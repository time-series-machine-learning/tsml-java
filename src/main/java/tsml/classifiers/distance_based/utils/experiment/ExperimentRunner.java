package tsml.classifiers.distance_based.utils.experiment;

public class ExperimentRunner {
    public static void main(String[] args) throws Exception {
        Experiment.main(
                "-r", "results"
                , "-d", "/bench/phd/data/all"
                , "-p", "GunPoint"
                , "-c", "PF_R5_OOB"
//                , "-c", "PF_R5"
                , "-s", "0"
                , "-l", "all"
                , "-o"
                , "--cp"
                , "--cpi", "30s"
                , "--rcp"
                , "-e"
//                , "--ttl", "10s"
//                , "--ttl", "20s"
//                , "--ttl", "30s"
                , "--ttl", "1m"
                , "--ttl", "2m"
                , "--ttl", "3m"
                , "--ttl", "4m"
        );
    }
}
