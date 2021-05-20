package tsml.classifiers.shapelet_based.dev.type;

public abstract class ShapeletSingle extends ShapeletMV{

    protected int start;

    protected int instanceIndex;




    public ShapeletSingle(int start, int length, int instanceIndex, double classIndex){
        super(length,classIndex);
        this.start = start;
        this.instanceIndex = instanceIndex;
    }

    public int getStart(){
        return start;
    }

    public int getInstanceIndex(){
        return instanceIndex;
    }


}
