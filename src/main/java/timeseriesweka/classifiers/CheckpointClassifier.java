/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package timeseriesweka.classifiers;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

/**
 * Interface that allows the user to allow a classifier to checkpoint, i.e. 
save its current state and then load it again to continue building the model on 
a separate run.

By default this involves simply saving and loading a serialised the object 

known classifiers: none

Requires two methods 
number 

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
            out.close();
            fos.close();
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
