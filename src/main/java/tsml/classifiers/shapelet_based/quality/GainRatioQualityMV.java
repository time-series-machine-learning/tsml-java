package tsml.classifiers.shapelet_based.quality;

import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.classifiers.shapelet_based.distances.ShapeletDistanceMV;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.core.DenseInstance;
import weka.core.Instances;

public class GainRatioQualityMV extends ShapeletQualityMV {

    public GainRatioQualityMV(double[][][] instancesArray,
                              int[] classIndexes,
                              String[] classNames,
                              int[] classCounts,
                              ShapeletDistanceMV distance){
       super(instancesArray,classIndexes,classNames,classCounts,distance);
    }

    @Override
    public double calculate(ShapeletMV candidate) {

        this.instances = new Instances("TestInstances",atts,0);
        this.instances.setClassIndex(1);

        for (int i=0;i< instancesArray.length;i++){
            double dist = distance.calculate(candidate,instancesArray[i]);
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
