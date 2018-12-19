/*
 * Sakoe Chiba DTW distance metric
 */
package timeseriesweka.elastic_distance_measures;

import weka.core.Instances;

/**
 *
 * @author Chris Rimmer
 */
public class SakoeChibaDTW extends BasicDTW {

//    private int bandSize;

    private double bandPercent;

    /**
     * Creates new Sakoe Chiba Distance metric
     * 
     * @param bandPercent warping window width as a percentage 
     * @throws IllegalArgumentException bandSize must be > 0
     */
    public SakoeChibaDTW(double bandPercent) throws IllegalArgumentException {
        super();
        setup(bandPercent);
    }


    /**
     * sets up the distance metric
     * 
     * @param bandSize
     * @throws IllegalArgumentException 
     */
//    private void setup(int bandSize) throws IllegalArgumentException {
//        if (bandSize < 1) {
//            throw new IllegalArgumentException("Band Size must be 1 or greater");
//        }
//
//        this.bandSize = bandSize;
//    }
    private void setup(double bandPercent) throws IllegalArgumentException {
        if (bandPercent <0 || bandPercent > 1) {
            throw new IllegalArgumentException("Band Size must be between 0 and 1");
        }

        this.bandPercent = bandPercent;
    }




    public int calculateBandSize(int instanceLength){
        if(this.bandPercent==0){
            return 1;
        }else{
            double width = instanceLength*this.bandPercent;
            return (int)Math.ceil(width);
        }
    }


    /**
     * calculates the distance between two instances (been converted to arrays)
     * 
     * @param first instance 1 as array
     * @param second instance 2 as array
     * @param cutOffValue used for early abandon
     * @return distance between instances
     */
    @Override
    public double distance(double[] first, double[] second, double cutOffValue) {

        int bandSize = this.calculateBandSize(first.length);

        //create empty array
        this.distances = new double[first.length][second.length];

        //first value
        this.distances[0][0] = (first[0] - second[0]) * (first[0] - second[0]);


        //top row
        for (int i = 1; i < second.length; i++) {
            if (i < bandSize) {
                this.distances[0][i] = this.distances[0][i - 1] + ((first[0] - second[i]) * (first[0] - second[i]));
            } else {
                this.distances[0][i] = Double.MAX_VALUE;
            }
        }

        //first column
        for (int i = 1; i < first.length; i++) {
            if (i < bandSize) {
                this.distances[i][0] = this.distances[i - 1][0] + ((first[i] - second[0]) * (first[i] - second[0]));
            } else {
                this.distances[i][0] = Double.MAX_VALUE;
            }
        }

        //warp rest
        double minDistance;
        
        // edited by Jay (07/07/15) - cutoff wasn't being used, so added overFlow etc to use early abandon
        
        boolean overFlow;
        for (int i = 1; i < first.length; i++) {
            overFlow = true;
            for (int j = 1; j < second.length; j++) {
                //Checks if i and j are within the band window
                if (i < j + bandSize && j < i + bandSize) {
                    minDistance = Math.min(this.distances[i][j - 1], Math.min(this.distances[i - 1][j], this.distances[i - 1][j - 1]));
                    //Assign distance
                    this.distances[i][j] = minDistance + ((first[i] - second[j]) * (first[i] - second[j]));
                } else {
                    this.distances[i][j] = Double.MAX_VALUE;
                }
                if(overFlow && this.distances[i][j] < cutOffValue){
                    overFlow=false;
                }
            }
            if(overFlow){
                return Double.MAX_VALUE;
            }
        }
        
        return this.distances[first.length - 1][second.length - 1];
    }

    /**
     * Sets the size of the warping window
     * 
     * @param bandSize band width
     * @throws IllegalArgumentException 
     */
    public void setBandSize(int bandSize) throws IllegalArgumentException {
        setup(bandSize);
    }
    
    /**
     * Gets the current warping window width
     * 
     * @return warping window width
     */
    public double getBandPercentage() {
        return this.bandPercent;
    }

    @Override
    public String toString() {
        return "SakoeChibaDTW{ " + "bandSize=" + this.bandPercent + "}";
    }
}