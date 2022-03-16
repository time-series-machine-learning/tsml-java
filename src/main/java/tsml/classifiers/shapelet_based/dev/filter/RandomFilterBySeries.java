package tsml.classifiers.shapelet_based.dev.filter;

import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.classifiers.selection.ElbowSelection;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctionsIndependent;
import tsml.classifiers.shapelet_based.dev.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.dev.type.ShapeletIndependentMV;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class RandomFilterBySeries extends RandomFilter {

    class ShapeletGroup implements Comparable<ShapeletGroup>{
        private List<ShapeletMV> shapelets;
        private int dimension,classIndex;
        private double quality;
        public ShapeletGroup( int dimension,int classIndex){
            shapelets = new ArrayList<>();
            this.dimension = dimension;
            this.classIndex = classIndex;
        }
        public void setQuality(double q){
            this.quality = q;
        }
        public void setShapelets( List<ShapeletMV> shapelets){
            this.shapelets = shapelets;
            this.setQuality(this.shapelets.stream().mapToDouble(ShapeletMV::getQuality).average().orElse(0));
        }

        public String toString(){
            return "Dimension " + dimension + " class " + classIndex + " quality " + quality;
        }

        public int compareTo(ShapeletGroup other) {
            return (Double.compare(other.quality, this.quality));
        }
    }

    @Override
    public ArrayList<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instances) {

        long start = System.nanoTime();


        params.k = 100;
        params.maxIterations = 100000;
        params.contractTimeHours = 1;



        ShapeletFunctions fun = params.type.createShapeletType();
        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances);

        ArrayList<ShapeletGroup> shapeletGroup = new ArrayList<ShapeletGroup>();
        RandomFilter randomFilter = new RandomFilter();
        randomFilter.setHourLimit(1);

        for (int i=0;i<instances.getMaxNumDimensions();i++){
            for (int j=0;j<instances.numClasses();j++){
                final int finalJ = j;
                int[] indexes = IntStream.range(0, instances.numInstances())
                        .filter(index -> instances.get(index).getLabelIndex() == finalJ)
                        .toArray();
                ShapeletGroup sg = new ShapeletGroup(i,j);

                sg.setShapelets(randomFilter.findShapelets(params,instances,indexes,i));
                System.out.println(sg);
                shapeletGroup.add(sg);

            }

        }

        Set<Integer> selections = new HashSet<>();
        for (int j=0;j<instances.numClasses();j++){
            final int localJ = j;
            List<ShapeletGroup> elements = shapeletGroup.stream().filter(sg -> sg.classIndex == localJ).collect(Collectors.toList());
            Collections.sort(elements);
            int elbow = getElbow((ArrayList<ShapeletGroup>) elements);
            for (int i=0;i<elbow+1;i++){
                selections.add(elements.get(i).dimension);
            }
            System.out.println(selections);
        }
        System.out.println(selections);

        return null;


    }

    protected int getElbow(ArrayList<ShapeletGroup> dimensionResults){
        int  nPoints = dimensionResults.size();

        ShapeletGroup firstPoint = dimensionResults.get(0);
        ShapeletGroup lastPoint = dimensionResults.get(nPoints-1);

        double[] lineVec = {nPoints-1, lastPoint.quality- firstPoint.quality};
        double lineVecSumSqrt = Math.sqrt(lineVec[0]*lineVec[0]+lineVec[1]*lineVec[1]);
        double[] lineVecNorm = {lineVec[0] / lineVecSumSqrt, lineVec[1] /lineVecSumSqrt};
        double[][] vecFromFirst = new double[nPoints][2];
        double[][] scalar = new double[nPoints][2];
        double[][] vecFromFirstParallel = new double[nPoints][2];
        double[][] vec_to_line = new double[nPoints][2];

        double[] scalarProd = new double[nPoints];
        double[] distToLine = new double[nPoints];
        int index = 0;
        double maxDistToLine = -9999;
        for (int i=0;i<nPoints;i++){
            vecFromFirst[i][0] = i;
            vecFromFirst[i][1] = dimensionResults.get(i).quality - firstPoint.quality;
            scalar[i][0] = vecFromFirst[i][0] * lineVecNorm[0];
            scalar[i][1] = vecFromFirst[i][1] * lineVecNorm[1];
            scalarProd[i] =  scalar[i][0] + scalar[i][1];
            vecFromFirstParallel[i][0] = scalarProd[i] * lineVecNorm[0];
            vecFromFirstParallel[i][1] = scalarProd[i] * lineVecNorm[1];
            vec_to_line[i][0] = vecFromFirst[i][0] - vecFromFirstParallel[i][0];
            vec_to_line[i][1] = vecFromFirst[i][1] - vecFromFirstParallel[i][1];
            distToLine[i] = Math.sqrt(  vec_to_line[i][0]*vec_to_line[i][0] + vec_to_line[i][1]*vec_to_line[i][1] );
            if (distToLine[i]>maxDistToLine){
                maxDistToLine = distToLine[i];
                index = i;
            }
        }
        return index;
    }




}
