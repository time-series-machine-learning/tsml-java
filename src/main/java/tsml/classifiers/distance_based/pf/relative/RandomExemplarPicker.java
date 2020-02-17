package tsml.classifiers.distance_based.pf.relative;

import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class RandomExemplarPicker implements ExemplarPicker, Randomizable {

    private int seed = 0;
    private Random random = new Random(seed);

    @Override public List<Instance> pickExemplars(final Instances data) {
        List<Instance> exemplars = new ArrayList<>();
        Map<Double, Instances> instancesByClass = Utilities.instancesByClass(data);
        for(int i = 0; i < data.numClasses(); i++) {
            Instances instances = instancesByClass.get(i);
            Instance exemplar = instances.get(random.nextInt(instances.size()));
            exemplars.add(exemplar);
        }
        return exemplars;
    }

    @Override
    public void setSeed(int seed) {
        this.seed = seed;
        random.setSeed(seed);
    }

    @Override
    public int getSeed() {
        return seed;
    }
}
