package tsml.classifiers.distance_based.interval;

import weka.core.Instances;

import java.util.List;

public interface Split {
    List<Instances> getParts();
    Instances getData();
    default void cleanUp() {

    }
    double getScore();
    void setScore(double score);
    List<Instances> split(Instances data);


//    private Instances data;
//    private List<Instances> parts = new ArrayList<>();
//    private double score = -1;
//
//    public Split() {
//
//    }
//
//    public List<Instances> getParts() {
//        return parts;
//    }
//
//    public void setParts(final List<Instances> parts) {
//        this.parts = parts;
//    }
//
//    public double getScore() {
//        return score;
//    }
//
//    public void setScore(final double score) {
//        this.score = score;
//    }
//
//    @Override public String toString() {
//        return "Split{" +
//            "numParts=" + parts.size() +
//            ", score=" + score +
//            '}';
//    }
//
//    public Instances getData() {
//        return data;
//    }
//
//    public void setData(final Instances data) {
//        this.data = data;
//    }
}
