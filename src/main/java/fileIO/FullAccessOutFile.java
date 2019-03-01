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
package fileIO;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.util.TreeSet;

public class FullAccessOutFile extends OutFile{

    public FullAccessOutFile(String n) {
        super(n);
    }
    @Override
    public void closeFile() {
        outFile.close();
        File f=new File(name);
        TreeSet<PosixFilePermission> perms=new TreeSet<>();
        for(PosixFilePermission p:PosixFilePermission.values())
        perms.add(p);
        Path path=f.toPath();
        try{
            Files.setPosixFilePermissions(path, perms);
        }catch(Exception e){
            System.out.println("UNABLE TO CHANGE PERMISSIONS FOR FILE "+name);
        }

    }
}
	