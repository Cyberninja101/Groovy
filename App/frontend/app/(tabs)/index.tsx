import React, { useEffect, useState } from "react";
import { View, Text, Pressable, StyleSheet, ActivityIndicator, Alert } from "react-native";
import * as DocumentPicker from "expo-document-picker";
import { Picker } from "@react-native-picker/picker";

type PickedFile = {
  uri: string;
  name: string;
  mimeType?: string;
  size?: number;
};

type UploadItem = {
  id: string;
  name: string;
  ts?: number;
};

const BACKEND_BASE = "http://10.48.46.35:8000";

export default function HomeScreen() {
  const [file, setFile] = useState<PickedFile | null>(null);
  const [uploading, setUploading] = useState(false);

  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [selectedUploadId, setSelectedUploadId] = useState<string>("");

  async function loadUploads() {
    try {
      const resp = await fetch(`${BACKEND_BASE}/uploads`);
      if (!resp.ok) throw new Error(`Failed to load uploads (${resp.status})`);
      const data = await resp.json();
      const items: UploadItem[] = data.items ?? [];
      setUploads(items);

      // auto-select most recent if none selected
      if (!selectedUploadId && items.length > 0) {
        setSelectedUploadId(items[0].id);
      }
    } catch (e: any) {
      // don’t hard-fail the UI; just show a warning
      console.log(e?.message ?? e);
    }
  }

  useEffect(() => {
    loadUploads();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function pickMp3() {
    const res = await DocumentPicker.getDocumentAsync({
      type: ["audio/mpeg", "audio/*"],
      copyToCacheDirectory: true,
      multiple: false,
    });

    if (res.canceled) return;

    const asset = res.assets?.[0];
    if (!asset) return;

    if (asset.name && !asset.name.toLowerCase().endsWith(".mp3")) {
      Alert.alert("Please choose an MP3 file.");
      return;
    }

    setFile({
      uri: asset.uri,
      name: asset.name ?? "audio.mp3",
      mimeType: asset.mimeType,
      size: asset.size,
    });
  }

  async function uploadToBackend() {
    if (!file) return Alert.alert("Pick an MP3 first.");

    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", {
        uri: file.uri,
        name: file.name,
        type: file.mimeType ?? "audio/mpeg",
      } as any);

      const resp = await fetch(`${BACKEND_BASE}/upload`, {
        method: "POST",
        body: form,
      });

      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(`Upload failed (${resp.status}). ${text}`);
      }

      const data = await resp.json();
      Alert.alert("Uploaded!", data?.name ?? "Success");

      // refresh dropdown items + auto-select the new one
      await loadUploads();
      if (data?.id) setSelectedUploadId(data.id);

      setFile(null);
    } catch (e: any) {
      Alert.alert("Error", e?.message ?? String(e));
    } finally {
      setUploading(false);
    }
  }

  const selectedName = uploads.find((u) => u.id === selectedUploadId)?.name ?? "";

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Upload MP3</Text>

      <Pressable style={styles.button} onPress={pickMp3} disabled={uploading}>
        <Text style={styles.buttonText}>Choose MP3</Text>
      </Pressable>

      {file && (
        <View style={styles.card}>
          <Text numberOfLines={1} style={styles.filename}>{file.name}</Text>
          <Text style={styles.meta}>{file.size ? `${Math.round(file.size / 1024)} KB` : ""}</Text>
        </View>
      )}

      <Pressable
        style={[styles.button, (!file || uploading) && styles.buttonDisabled]}
        onPress={uploadToBackend}
        disabled={!file || uploading}
      >
        {uploading ? <ActivityIndicator /> : <Text style={styles.buttonText}>Submit</Text>}
      </Pressable>

      <Text style={styles.sectionTitle}>Previously uploaded</Text>

      <View style={styles.pickerWrap}>
        <Picker
          selectedValue={selectedUploadId}
          onValueChange={(val) => setSelectedUploadId(String(val))}
        >
          <Picker.Item label="Select an upload..." value="" />
          {uploads.map((u) => (
            <Picker.Item key={u.id} label={u.name} value={u.id} />
          ))}
        </Picker>
      </View>

      {selectedUploadId ? (
        <Text style={styles.note}>Selected: {selectedName}</Text>
      ) : (
        <Text style={styles.note}>No upload selected</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, gap: 14, justifyContent: "center" },
  title: { fontSize: 28, fontWeight: "700", textAlign: "center" },
  sectionTitle: { marginTop: 10, fontSize: 18, fontWeight: "600" },
  button: { padding: 14, borderRadius: 12, alignItems: "center", backgroundColor: "#2d6cdf" },
  buttonText: { color: "white", fontSize: 16, fontWeight: "600" },
  buttonDisabled: { opacity: 0.5 },
  card: { padding: 12, borderRadius: 12, backgroundColor: "#eee" },
  filename: { fontSize: 16, fontWeight: "600" },
  meta: { marginTop: 4, opacity: 0.7 },
  pickerWrap: { borderRadius: 12, overflow: "hidden", backgroundColor: "#eee" },
  note: { marginTop: 6, textAlign: "center", opacity: 0.8 },
});