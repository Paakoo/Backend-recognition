import h5py
import numpy as np

file_path = "model/ds_model_face.h5"
output_txt_path = "output.txt"

# Membuka file HDF5 dalam mode read-only
with h5py.File(file_path, "r") as h5_file:
    with open(output_txt_path, 'w') as txt_file:
        # Fungsi untuk mencetak atribut dan data dari setiap objek
        def print_attrs(name, obj):
            txt_file.write(f"{name}:\n")
            for key, val in obj.attrs.items():
                txt_file.write(f"    {key}: {val}\n")
            if isinstance(obj, h5py.Dataset):
                txt_file.write("    Data:\n")
                np.savetxt(txt_file, obj[()], fmt='%s')
            txt_file.write("\n")

        # Kunjungi setiap item dalam file dan cetak atribut serta datanya
        h5_file.visititems(print_attrs)

print(f"Data has been written to {output_txt_path}")

# print(list(h5_file.keys()))