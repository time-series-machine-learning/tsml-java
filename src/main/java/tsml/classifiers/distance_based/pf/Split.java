package tsml.classifiers.distance_based.pf;

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
    void setData(Instances data);
    List<Instances> split();
}
