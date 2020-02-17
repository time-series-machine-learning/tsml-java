package tsml.classifiers.distance_based.pf.relative;

import com.google.common.collect.Lists;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.DistanceMeasureConfigs;
import tsml.classifiers.distance_based.pf.Scorer;
import tsml.classifiers.distance_based.pf.Split;
import tsml.classifiers.distance_based.pf.Splitter;
import utilities.Resetable;
import utilities.params.ParamSet;
import utilities.params.ParamSpace;
import utilities.serialisation.SerConsumer;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

import java.util.*;

public class RandomExemplarSplitter implements Splitter, Resetable, Randomizable {

    private int seed = 0;
    private Random random = new Random(seed);
    private List<ParamSpace> paramSpaces;
    private Scorer scorer = Scorer.giniScore;
    private ExemplarPicker picker = null;
    private SerConsumer<Instances> splitSetupFunction = (SerConsumer<Instances>) instances -> {
        RandomExemplarPicker randomExemplarPicker = new RandomExemplarPicker();
        randomExemplarPicker.setSeed(random.nextInt());
        picker = randomExemplarPicker;
        paramSpaces = Lists.newArrayList(
                DistanceMeasureConfigs.buildDtwSpaceV2(instances),
                DistanceMeasureConfigs.buildDdtwSpaceV2(instances),
                DistanceMeasureConfigs.buildWdtwSpaceV2(),
                DistanceMeasureConfigs.buildWddtwSpaceV2(),
                DistanceMeasureConfigs.buildLcssSpace(instances),
                DistanceMeasureConfigs.buildErpSpace(instances),
                DistanceMeasureConfigs.buildTwedSpace(),
                DistanceMeasureConfigs.buildMsmSpace()
        );
    };
    private boolean reset = true;

    @Override public ExemplarSplit split(final Instances data) {
        if(reset) {
            splitSetupFunction.accept(data);
            reset = false;
        }
        ParamSpace paramSpace = paramSpaces.get(random.nextInt(paramSpaces.size()));
        ParamSet paramSet = paramSpace.get(random.nextInt(paramSpace.size()));
        List<Object> distanceFunctions = paramSet.get(DistanceMeasure.DISTANCE_FUNCTION_FLAG);
        if(distanceFunctions.size() != 1) {
            throw new IllegalStateException("was expecting only 1");
        }
        DistanceFunction distanceFunction = (DistanceFunction) distanceFunctions.get(0);
        ExemplarSplit split = new ExemplarSplit();
        split.setSeed(random.nextInt());
        split.setDistanceFunction(distanceFunction);
        List<Instance> exemplars = picker.pickExemplars(data);
        split.setExemplars(exemplars);
        List<Instances> parts = split.split(data);
        double score = scorer.findScore(data, parts);
        split.setScore(score);
        return split;
    }

    @Override public boolean isReset() {
        return reset;
    }

    @Override public void setReset(final boolean state) {
        reset = state;
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
