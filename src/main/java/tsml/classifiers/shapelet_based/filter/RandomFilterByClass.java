package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.shapelet_based.classifiers.MSTC;
import tsml.classifiers.shapelet_based.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;

public class RandomFilterByClass extends RandomFilter {


    @Override
    public ArrayList<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instances) {
        long start = System.nanoTime();
        int numClasses = instances.getClassCounts().length;
        ShapeletFunctions type = params.type.createShapeletType();
        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances,
                params.distance.createShapeletDistance());


        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();

        for (int i=0;i<numClasses;i++){
            System.out.println("Class "  + i);
            ArrayList<Integer> options = new ArrayList<Integer>();
            for (int j=0;j<instances.numInstances();j++){
                if (instances.get(j).getTargetValue()==i)
                    options.add(j);
            }

            ArrayList<ShapeletMV> classShapelets = randomShapelets(params,instances,options,start, type, quality, params.maxIterations/numClasses, params.k/numClasses);
        //    System.out.println(classShapelets);
            shapelets.addAll(classShapelets);
        }


        return shapelets;
    }



}
