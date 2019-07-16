package utilities;

public interface Copyable {

    Object copy() throws
                  Exception;

    void copyFrom(Object object) throws Exception;

}
