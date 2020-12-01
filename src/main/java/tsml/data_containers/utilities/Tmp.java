package tsml.data_containers.utilities;

import experiments.data.DatasetLoading;
import weka.core.Instances;

public class Tmp {
    public static void main(String[] args) throws Exception {
        final Instances[] instances = DatasetLoading.sampleBasicMotions(0);
        System.out.println(instances);
    }
}
