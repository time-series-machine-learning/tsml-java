package tsml.classifiers.distance_based.ee;

import com.google.common.collect.Lists;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.distances.DistanceMeasureConfigs;
import utilities.params.ParamSet;
import utilities.params.ParamSpace;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class ET extends EnhancedAbstractClassifier implements TrainTimeContractable {

    private boolean rebuild = true;
    private List<Function<Instances, ParamSpace>> paramSpaceBuilders = Lists.newArrayList(
            DistanceMeasureConfigs::buildDtwSpaceV2,
            DistanceMeasureConfigs::buildDdtwSpaceV2,
            i -> DistanceMeasureConfigs.buildWdtwSpaceV2(),
            i -> DistanceMeasureConfigs.buildWddtwSpaceV2(),
            DistanceMeasureConfigs::buildErpSpace,
            DistanceMeasureConfigs::buildLcssSpace,
            i -> DistanceMeasureConfigs.buildMsmSpace(),
            i -> DistanceMeasureConfigs.buildTwedSpace()
    );
    private List<ParamSpace> paramSpaces = new ArrayList<>();
    private long trainTimeLimitNanos = -1;
    private List<List<Double>> transformedTrainData;

    @Override
    public void setRebuild(boolean rebuild) {
        this.rebuild = rebuild;
        super.setRebuild(rebuild);
    }

    @Override
    public void buildClassifier(Instances trainData) throws Exception {
        super.buildClassifier(trainData);
        if(rebuild) {
            for(Function<Instances, ParamSpace> builder : paramSpaceBuilders) {
                ParamSpace paramSpace = builder.apply(trainData);
                paramSpaces.add(paramSpace);
            }
            transformedTrainData = new ArrayList<>();
            for(Instance instance : trainData) {
                transformedTrainData.add(Lists.newArrayList(instance.classValue()));
            }
        }
        while (hasRemainingTraining()) {
            // pick a random distance measure and index
            ParamSpace paramSpace = paramSpaces.get(rand.nextInt(paramSpaces.size()));
            ParamSet paramSet = paramSpace.get(rand.nextInt(paramSpace.size()));
            List<Object> distanceMeasures = paramSet.get(DistanceMeasure.DISTANCE_FUNCTION_FLAG);
            if(distanceMeasures.size() != 1) {
                throw new IllegalStateException("there shouldn't be more than 1 distance measure");
            }
            DistanceMeasure distanceMeasure = (DistanceMeasure) distanceMeasures.get(0);
            distanceMeasure.setInstances(trainData);
            // todo check distance measure params are set
            // todo track longest time for prediction
            // todo resource monitors
            Instance exemplar = trainData.get(rand.nextInt(trainData.size()));
            for(int i = 0; i < trainData.size(); i++) {
                Instance instance = trainData.get(i);
                double distance = distanceMeasure.distance(instance, exemplar);
                List<Double> transformedInstance = transformedTrainData.get(i);
                transformedInstance.add(transformedInstance.size() - 1, distance);
            }
        }
    }

    @Override
    public void setTrainTimeLimitNanos(long trainTimeLimitNanos) {
        this.trainTimeLimitNanos = trainTimeLimitNanos;
    }

    @Override
    public long getTrainTimeLimitNanos() {
        return trainTimeLimitNanos;
    }
}
