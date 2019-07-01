/* 
 * Piecewise DTW distance metric 
 */
package timeseriesweka.elastic_distance_measures;

/**
 *
 * @author Chris Rimmer
 */
public class PiecewiseDTW extends BasicDTW {
    
    private int frameSize;
    private double[] reducedDimensionSeries1;
    private double[] reducedDimensionSeries2;
    
    /**
     * Creates new Piecewise DTW distance metric
     * 
     * @param frameSize size of frame to split the data
     * @throws IllegalArgumentException frameSize must be a factor of number of attributes in data
     */
    public PiecewiseDTW(int frameSize) throws IllegalArgumentException{
        super();
        setup(frameSize);
    }
    
    
    /**
     * Setup distance metric
     * 
     * @param frameSize size of frame to split the data
     * @throws IllegalArgumentException frameSize must be a factor of number of attributes in data
     */
    private void setup(int frameSize) throws IllegalArgumentException{
        if(frameSize < 1){
            throw new IllegalArgumentException("Frame Size must be 1 or greater");
        }
        
        this.frameSize = frameSize;
    }
    
    /**
     * reduces the data dimensionally in equal sized frames and passes to superclass to calculate distance
     * 
     * @param first array 1
     * @param second array 2 
     * @param cutOffValue used for early abandon
     * @return distance between instances
     */
    @Override
    public double distance(double[] first,double[] second, double cutOffValue){
        //check can divide into equal parts
        if(first.length % this.frameSize != 0){
            throw new IllegalArgumentException("Frame size must be a factor of the number of attributes");
        }
        
        //setup arrays
        int seriesLength = first.length/this.frameSize;
        this.reducedDimensionSeries1 = new double[seriesLength];
        this.reducedDimensionSeries2 = new double[seriesLength];
        
        double series1Frame = 0;
        double series2Frame = 0;
        
        //reduces the dimensionality of the data
        for(int i = 0, reducedPos = 0; i < first.length; i+=this.frameSize, reducedPos++){
            series1Frame = 0;
            series2Frame = 0;
            for(int j = i; j < i+this.frameSize; j++){
                series1Frame += first[j];
                series2Frame += second[j];
            }
            this.reducedDimensionSeries1[reducedPos] = series1Frame/this.frameSize;
            this.reducedDimensionSeries2[reducedPos] = series2Frame/this.frameSize;
        }
                
        return super.distance(reducedDimensionSeries1, reducedDimensionSeries2, cutOffValue);
        
    }

    /**
     * Sets the frame size
     * 
     * @param frameSize size of frame to split the data
     * @throws IllegalArgumentException frameSize must be a factor of number of attributes in data
     */
    public void setFrameSize(int frameSize) throws IllegalArgumentException{
        setup(frameSize);
    }

    /**
     * Gets the current frame size
     * 
     * @return current frame size
     */
    public int getFrameSize() {
        return frameSize;
    }

    /**
     * Gets dimensionally reduced series 1
     * 
     * @return reduced series 1
     */
    public double[] getReducedDimensionSeries1() {
        return reducedDimensionSeries1;
    }

    /**
     * Gets dimensionally reduced series 2
     * 
     * @return reduced series 2
     */
    public double[] getReducedDimensionSeries2() {
        return reducedDimensionSeries2;
    }
    
    /**
     * Prints the reduced dimensionality series arrays
     */
    public void printReducedSeries(){        
        System.out.println("------------------ Reduced Series 1 ------------------");
        for(int i = 0; i<this.reducedDimensionSeries1.length; i++){
            System.out.print(" "+ reducedDimensionSeries1[i]+"\n");
        }
        System.out.println("------------------ End ------------------");
        System.out.println("------------------ Reduced Series 2 ------------------");
        for(int i = 0; i<this.reducedDimensionSeries2.length; i++){
            System.out.print(" "+ reducedDimensionSeries2[i]+"\n");
        }
        System.out.println("------------------ End ------------------");
    }

    @Override
    public String toString() {
        return "PiecewiseDTW{ " + "frameSize=" + this.frameSize + ", }";
    }
}