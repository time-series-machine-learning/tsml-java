package tsml.classifiers.distance_based.utils.classifiers;

import experiments.Experiments;

public class ExperimentRunner {
    public static void main(String[] args) throws Exception {
        Experiments.main(new String[] {
                "-dp", "/bench/phd/data/all"
                , "-rp", "results"
                , "-cn", "PF_R5_OOB"
                , "-dn", "SyntheticControl"
                , "-f", "1"
                , "-cp", "true"
                , "-tb", "true"
                , "-ctr", "30s"
                , "-gtf", "true"
                , "-l", "all"
                , "--force", "true"
                , "-ectr", "true"
        });
    }
}
