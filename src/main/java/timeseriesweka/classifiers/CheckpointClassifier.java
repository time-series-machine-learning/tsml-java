/*
  Interface that allows the user to allow a classifier to checkpoint, i.e. 
save its current state and then load it again to continue building the model on 
a separate run.

By default this involves simply saving and loading a serialised the object 

known classifiers: none

Requires two methods 
number 

*/
package timeseriesweka.classifiers;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

/**
 *
 * @author Tony Bagnall 2018
 */
public interface CheckpointClassifier extends Serializable{

    //Set the path where checkpointed versions will be stored
    public void setSavePath(String path);
    //Define how to copy from a loaded object to this object
    public void copyFromSerObject(Object obj) throws Exception;

    //Override both if not using Java serialisation    
    public default void saveToFile(String filename) throws IOException{
        FileOutputStream fos =
        new FileOutputStream(filename);
        try (ObjectOutputStream out = new ObjectOutputStream(fos)) {
            out.writeObject(this);
            fos.close();
            out.close();
        }
    }
    public default void loadFromFile(String filename) throws Exception{
        FileInputStream fis = new FileInputStream(filename);
        try (ObjectInputStream in = new ObjectInputStream(fis)) {
            Object obj=in.readObject();
            copyFromSerObject(obj);
        }
    }
    
    
}
