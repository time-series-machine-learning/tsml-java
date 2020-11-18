package tsml.classifiers.distance_based.utils.experiment;

public class ExperimentRunner {
    public static void main(String[] args) throws Exception {
        Experiment.main(
                "-r", "results"
                , "-d", "/bench/phd/data/all"
                , "-p", "GunPoint"
                , "-c", "PF_R5"
                , "-s", "0"
                , "-l", "all"
                , "--ttl", "1h30m"
                , "--ttl", "2m10s"
                , "--ttl", "90m"
                , "--ttl", "130s"
                , "--ttl", "1m70s"
        );
    }
}
