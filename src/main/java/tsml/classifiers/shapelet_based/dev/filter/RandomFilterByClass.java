package tsml.classifiers.shapelet_based.dev.filter;

import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.List;

public class RandomFilterByClass extends RandomFilter {


    @Override
    public ArrayList<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instances) {
        this.start = System.nanoTime();

        int numClasses = instances.getClassCounts().length;
        ShapeletFunctions type = params.type.createShapeletType();
        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances);


        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();

        TimeSeriesInstances[] instancesArray = new TimeSeriesInstances[numClasses];

        for (int i=0;i<numClasses;i++){
            instancesArray[i] = new TimeSeriesInstances(instances.getClassLabels());

        }
        for (int j=0;j<instances.numInstances();j++){
            instancesArray[instances.get(j).getLabelIndex()].add(instances.get(j));
        }

        RandomFilter filter = new RandomFilter();
        filter.setHourLimit(params.contractTimeHours);
        MSTC.ShapeletParams classParams = new MSTC.ShapeletParams(params);
        classParams.k = params.k / numClasses;
        classParams.maxIterations = params.maxIterations / numClasses;
        for (int i=0;i<numClasses;i++){

              System.out.println("Shapelets for class " + i + "/" + numClasses);

        //    System.out.println(classShapelets);
            List<ShapeletMV> classShapelets = filter.findShapelets(classParams, quality, instancesArray[i]);
            shapelets.addAll(classShapelets);
        }


        return shapelets;
    }



}
