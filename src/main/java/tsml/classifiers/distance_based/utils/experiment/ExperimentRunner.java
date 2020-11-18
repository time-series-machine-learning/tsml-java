package tsml.classifiers.distance_based.utils.experiment;

public class ExperimentRunner {
    public static void main(String[] args) throws Exception {
        Experiment.main(
                "-r", "results"
                , "-d", "/bench/data"
                , "-p", "ElectricDevices"
                , "-c", "KNN_LOOCV_R"
                , "-s", "0"
                , "-l", "all"
                , "--trainTimeContract", "10", "s"
                , "--trainTimeContract", "20", "s"
                , "--trainTimeContract", "30", "s"
        );
    }
}
