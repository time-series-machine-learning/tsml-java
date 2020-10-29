package tsml.classifiers.distance_based.utils.classifiers;

import experiments.Experiments;

public class ExperimentRunner {
    public static void main(String[] args) throws Exception {
        Experiments.main(new String[] {
                "-dp", "/bench/phd/data/all"
                , "-rp", "results"
                , "-cn", "PF_R5"
                , "-dn", "SyntheticControl"
                , "-f", "1"
                , "-cp", "true"
                , "-tb", "true"
                , "-ctr", "30s"
                , "-l", "all"
                , "--force", "true"
        });
    }
}
