package vector_classifiers;

/**
 *
 * @author ajb
 */
public interface SaveEachParameter {
    void setPathToSaveParameters(String r);
    default void setSaveEachParaAcc(){setSaveEachParaAcc(true);}
    void setSaveEachParaAcc(boolean b);
}
