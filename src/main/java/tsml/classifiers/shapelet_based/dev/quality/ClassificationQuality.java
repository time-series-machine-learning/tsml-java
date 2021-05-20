package tsml.classifiers.shapelet_based.dev.quality;

import tsml.classifiers.shapelet_based.dev.distances.ShapeletDistanceFunction;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;
import weka.classifiers.trees.DecisionStump;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.classifiers.Evaluation;

public class ClassificationQuality extends ShapeletQualityFunction {

    public ClassificationQuality(TimeSeriesInstances instances,
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


            try {
                DecisionStump dec = new DecisionStump();
                Evaluation  eval = new Evaluation ( this.instances);
                dec.buildClassifier(this.instances);
                eval.evaluateModel(dec, this.instances);
                return eval.pctCorrect();
            } catch (Exception e) {
                e.printStackTrace();

            }

            return 0;


    }


}
