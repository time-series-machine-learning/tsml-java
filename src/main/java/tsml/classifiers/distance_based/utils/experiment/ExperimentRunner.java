package tsml.classifiers.distance_based.utils.experiment;

public class ExperimentRunner {
    public static void main(String[] args) throws Exception {
        Experiment.main(
                "-r", "results"
                , "-d", "/bench/phd/data/all"
                , "-p", "GunPoint"
                , "-c", "PF"
//                , "-c", "PF_R5"
                , "-s", "0"
                , "-l", "all"
                , "-o"
                , "--cp"
                , "--cpi", "30s"
                , "--rcp"
                , "--ttl", "1m"
                , "--ttl", "2m"
                , "--ttl", "3m"
        );
    }
}
