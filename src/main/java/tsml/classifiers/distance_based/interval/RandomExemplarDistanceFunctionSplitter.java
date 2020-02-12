package tsml.classifiers.distance_based.interval;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import utilities.ArrayUtilities;
import utilities.Resetable;
import utilities.Utilities;
import utilities.collections.PrunedMultimap;
import utilities.params.ParamSet;
import utilities.params.ParamSpace;
import utilities.serialisation.SerConsumer;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class RandomExemplarDistanceFunctionSplitter implements Splitter, Resetable {

    private Random random = new Random(0);
    private List<ParamSpace> paramSpaces;
    private Scorer scorer = Utilities::giniScore;
    private ExemplarPicker picker = null;
    private SerConsumer<Instances> splitSetupFunction = (SerConsumer<Instances>) instances -> {
        RandomExemplarPicker randomExemplarPicker = new RandomExemplarPicker();
        randomExemplarPicker.setRandom(random);
        picker = randomExemplarPicker;
    };
    private boolean reset = true;

    @Override public Split split(final Instances data) {
        if(reset) {
            splitSetupFunction.accept(data);
            reset = false;
        }
        ParamSpace paramSpace = paramSpaces.get(random.nextInt(paramSpaces.size()));
        ParamSet paramSet = paramSpace.get(random.nextInt(paramSpace.size()));
        DistanceFunction distanceFunction;
        Map<Instance, Instances> splitByExemplarMap = new HashMap<>();
        ExemplarSplit split = new ExemplarSplit();
        split.setDistanceFunction(distanceFunction);
        List<Instance> exemplars = picker.pickExemplars(data);
        split.setExemplars(exemplars);
        List<Instances> parts = split.split(data);
        int[] partSizes = new int[parts.size()];
        for(int i = 0; i < parts.size(); i++) {
            int size = parts.get(i).size();
            partSizes[i] = size;
        }
        double score = scorer.findScore(data.size(), partSizes);
        split.setScore(score);
        return split;
    }

    public Random getRandom() {
        return random;
    }

    public void setRandom(final Random random) {
        this.random = random;
    }

    @Override public boolean isReset() {
        return reset;
    }

    @Override public void setReset(final boolean state) {
        reset = state;
    }
}
