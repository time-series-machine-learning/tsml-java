package utilities;

public interface Debugable {
    boolean isDebug();
    void setDebug(boolean state);
    default void enableDebug() {
        setDebug(true);
    }
    default void disableDebug() {
        setDebug(false);
    }
}
