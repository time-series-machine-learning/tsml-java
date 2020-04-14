package tsml.classifiers.distance_based.utils.experiments;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ExperimentRunner {

    public static void main(String[] args) throws Exception {
        ExperimentBatch.main(
            "-r", "results"
            , "-c", "PT_R1_GINI"
            , "-d", "GunPoint"
            , "--dd", "/bench/datasets"
            , "-s", "0"
            , "--op"
//            ,"-e"
        );
    }
}
