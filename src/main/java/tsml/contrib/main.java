package tsml.contrib;

import experiments.data.DatasetLoading;
import weka.core.Instances;

public class main {
    public static void main(String [] args) throws Exception {

        Instances[] data = DatasetLoading.sampleBasicMotions(0);

        for (Instances x : data) {
            System.out.println(x.toString());
        }

    }
}
