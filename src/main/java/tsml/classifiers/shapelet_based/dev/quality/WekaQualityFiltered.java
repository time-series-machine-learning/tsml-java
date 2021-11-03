package tsml.classifiers.shapelet_based.dev.quality;

import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.DenseInstance;
import weka.core.Instances;

public class WekaQualityFiltered extends WekaQuality {


    public WekaQualityFiltered(WekaEvaluatorFactory eval,
                               TransformedInstancesFactory transformedInstances,
                               TimeSeriesInstances instances){
       super(eval,transformedInstances,instances);
    }

    @Override
    public double calculate(ShapeletFunctions fun, ShapeletMV candidate) {

        Instances instances = new Instances("TestInstances",atts,0);
        instances.setClassIndex(instances.attribute("class").index());

        if (trainInstances.numInstances()>100){
            for (int i=0;i< 100;i++){
                double dist = fun.sDist(candidate,trainInstances.get(MSTC.RAND.nextInt(trainInstances.numInstances())));
                double classIndex = this.transformedInstances.getClassIndex(trainInstances.get(i).getTargetValue(),candidate.getClassIndex());
                instances.add(new DenseInstance(1,new double[]{dist,classIndex}));

            }
        }else{
            for (int i=0;i< trainInstances.numInstances();i++){
                double dist = fun.sDist(candidate,trainInstances.get(i));
                double classIndex = this.transformedInstances.getClassIndex(trainInstances.get(i).getTargetValue(),candidate.getClassIndex());
                instances.add(new DenseInstance(1,new double[]{dist,classIndex}));

            }
        }






            try {
               eval.buildEvaluator(instances);

                return ((AttributeEvaluator)eval).evaluateAttribute(0);
            }catch (IllegalArgumentException e) {
               // e.printStackTrace();
                return 0;

            }
            catch (Exception e) {
                e.printStackTrace();

            }

            return 0;


    }




}
