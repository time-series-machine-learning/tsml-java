package tsml.classifiers.shapelet_based.classifiers;

public class ShapeletMV {

    private int start,end;
    private int seriesIndex;
    private int instanceIndex;
    private double[][] data;
    private double quality;

    public ShapeletMV(int start, int end, int instanceIndex, int seriesIndex, double[][] instance){
        this.start = start;
        this.end = end;
        this.instanceIndex = instanceIndex;
        this.seriesIndex = seriesIndex;
        this.setData(instance);
    }

    public void setData(double[][] instance) {
        if (this.seriesIndex == -1){
            this.data = new double[end-start][];
            for (int i=0;i<end-start;i++){
                this.data[i] = new double[instance[i].length];
                for (int j=0;j<instance[i].length;j++){
                    this.data[i][j] = instance[start+i][j];
                }
            }
        }else{
            this.data = new double[end-start][1];
            for (int i=0;i<end-start;i++){
                this.data[i] = new double[1];
                    this.data[i][1] = instance[start+i][this.seriesIndex];
            }

        }
    }

    public void setQuality(double quality){
        this.quality = quality;
    }

    public double getQuality(){
        return this.quality;
    }
    public double[][] getData(){
        return this.data;
    }

    public int getLength(){
        return end - start;
    }
}
