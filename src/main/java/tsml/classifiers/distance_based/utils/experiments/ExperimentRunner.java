package tsml.classifiers.distance_based.utils.experiments;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ExperimentRunner {

    public static void main(String[] args) throws Exception {
        ExperimentBatch.main(
            "-r", "results",
            "-c", "ED_1NN",
            "-d", "GunPoint",
            "-dd", "/bench/datasets",
            "-s", "0",
            "-e"
        );
    }
}
