package tsml.classifiers.shapelet_based.dev.quality;

import tsml.classifiers.shapelet_based.dev.distances.ShapeletDistanceFunction;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.core.DenseInstance;
import weka.core.Instances;

public class GainRatioQuality extends ShapeletQualityFunction {

    public GainRatioQuality(TimeSeriesInstances instances,
                            ShapeletDistanceFunction distance){
       super(instances,distance);
    }

    @Override
    public double calculate(ShapeletMV candidate) {

        this.instances = new Instances("TestInstances",atts,0);
        this.instances.setClassIndex(1);

        for (int i=0;i< trainInstances.numInstances();i++){
            double dist = distance.calculate(candidate,trainInstances.get(i));
            instances.add(new DenseInstance(1,new double[]{dist,(double)classIndexes[i]}));

        }
       // System.out.println(instances);

        GainRatioAttributeEval eval = new   GainRatioAttributeEval();
            try {
                eval.buildEvaluator(instances);
                return eval.evaluateAttribute(0);
            } catch (Exception e) {
                e.printStackTrace();

            }

            return 0;


    }


}
