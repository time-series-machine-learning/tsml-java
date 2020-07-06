package tsml.classifiers.distance_based.proximity;

import experiments.data.DatasetLoading;
import tsml.classifiers.distance_based.distances.transformed.TransformDistanceMeasure;
import utilities.InstanceTools;
import utilities.Utilities;
import weka.core.Instances;

import java.util.List;
import java.util.Map;

public class Tmp {
    public static void main(String[] args) throws Exception {
        final Instances instances = DatasetLoading.loadGunPoint();
        long sum = 0;
        int repeats = 100;
        for(int i = 0; i < repeats; i++) {
            long time = System.nanoTime();
            final Map<Double, Instances> map = Utilities.instancesByClass(instances);
            time = System.nanoTime() - time;
            sum += time;
        }
        System.out.println(sum / repeats);
    }
}
