package utilities;

public interface Copyable {

    default Object shallowCopy() throws
                  Exception {
        throw new UnsupportedOperationException();
    }

    default void shallowCopyFrom(Object object) throws Exception {
        throw new UnsupportedOperationException();
    }

    default void deepCopyFrom(Object object) throws Exception {
        throw new UnsupportedOperationException();
    }

    default Object deepCopy() throws Exception {
        throw new UnsupportedOperationException();
    }

}
