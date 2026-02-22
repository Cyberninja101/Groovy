import React, { useState } from "react";
import {
  View,
  Text,
  Pressable,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from "react-native";
import * as DocumentPicker from "expo-document-picker";
import { Picker } from "@react-native-picker/picker";

const BACKEND_BASE = "http://10.48.46.35:8000";

type LocalUploadItem = {
  id: string;
  name: string;
  uri: string;
  ts: number;
};

export default function HomeScreen() {
  const [uploads, setUploads] = useState<LocalUploadItem[]>([]);
  const [selectedUploadId, setSelectedUploadId] = useState<string>("");
  const [uploading, setUploading] = useState(false);

  // -------------------------
  // Pick MP3 (no file copy)
  // -------------------------
  async function pickMp3() {
  const res = await DocumentPicker.getDocumentAsync({
    type: ["audio/mpeg", "audio/*"],
    copyToCacheDirectory: true,
    multiple: false,
  });

  if (res.canceled) return;
  const asset = res.assets?.[0];
  if (!asset) return;

  const name = asset.name ?? `audio_${Date.now()}.mp3`;
  if (!name.toLowerCase().endsWith(".mp3")) {
    Alert.alert("Please choose an MP3 file.");
    return;
  }

  setUploading(true);
  try {
    const form = new FormData();
    form.append("file", {
      uri: asset.uri,
      name,
      type: "audio/mpeg",
    } as any);

    const resp = await fetch(`${BACKEND_BASE}/upload`, {
      method: "POST",
      body: form,
      // NOTE: do NOT manually set Content-Type for FormData in RN fetch
    });

    const text = await resp.text();
    if (!resp.ok) throw new Error(`Upload failed (${resp.status}): ${text}`);

    const data = JSON.parse(text);

    // IMPORTANT: use the backend-generated id (uuid) from Flask
    const newItem: LocalUploadItem = {
      id: data.id,     // <-- backend UUID
      name: data.name, // original filename
      uri: asset.uri,
      ts: Date.now(),
    };

    setUploads((prev) => [newItem, ...prev]);
    setSelectedUploadId(data.id);

    Alert.alert(
      "Uploaded!",
      `Server ID: ${data.id}\nBeats: ${data.beats_detected}\nTempo: ${data.tempo_bpm.toFixed?.(1) ?? data.tempo_bpm}`
    );
  } catch (e: any) {
    Alert.alert("Upload error", e?.message ?? String(e));
  } finally {
    setUploading(false);
  }
}

  // -------------------------
  // Upload selected file
  // -------------------------
  async function submitToBackend() {
    if (!selectedUploadId) return Alert.alert("Select an upload first.");

    setUploading(true);
    try {
      const resp = await fetch(`${BACKEND_BASE}/submit/${selectedUploadId}`, {
        method: "POST",
      });

      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(`Submit failed (${resp.status}). ${text}`);
      }

      const data = await resp.json();
      Alert.alert("Sent to Pi!", JSON.stringify(data.pi_ack ?? data, null, 2));
    } catch (e: any) {
      Alert.alert("Error", e?.message ?? String(e));
    } finally {
      setUploading(false);
    }
  }

  const selectedName =
    uploads.find((u) => u.id === selectedUploadId)?.name ?? "";

  return (
    <View style={styles.container}>
      <Text style={styles.title}>MP3 Manager</Text>

      <Pressable style={styles.button} onPress={pickMp3} disabled={uploading}>
        <Text style={styles.buttonText}>Choose MP3</Text>
      </Pressable>

      <Text style={styles.sectionTitle}>Saved MP3s</Text>

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

      <Pressable
        style={[
          styles.button,
          (!selectedUploadId || uploading) && styles.buttonDisabled,
        ]}
        onPress={submitToBackend}
        disabled={!selectedUploadId || uploading}
      >
        {uploading ? (
          <ActivityIndicator color="white" />
        ) : (
          <Text style={styles.buttonText}>Submit</Text>
        )}
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, gap: 14, justifyContent: "center" },
  title: { fontSize: 28, fontWeight: "700", textAlign: "center" },
  sectionTitle: { marginTop: 10, fontSize: 18, fontWeight: "600" },
  button: {
    padding: 14,
    borderRadius: 12,
    alignItems: "center",
    backgroundColor: "#2d6cdf",
  },
  buttonText: { color: "white", fontSize: 16, fontWeight: "600" },
  buttonDisabled: { opacity: 0.5 },
  pickerWrap: {
    borderRadius: 12,
    overflow: "hidden",
    backgroundColor: "#eee",
  },
  note: { marginTop: 6, textAlign: "center", opacity: 0.8 },
});