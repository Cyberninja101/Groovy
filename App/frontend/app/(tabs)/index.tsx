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
  id: string;   // backend uuid
  name: string; // original filename
  uri: string;
  ts: number;
};

// Hardcoded songs shown in the SAME dropdown.
// label = what user sees
// key   = what backend expects as beatmap name (NO spaces)
const HARDCODED_SONGS = [
  { label: "stacyMom", key: "stacyMom" },
  { label: "seven nation army", key: "seven_nation_army" },
  { label: "APT", key: "APT" },
];

export default function HomeScreen() {
  const [uploads, setUploads] = useState<LocalUploadItem[]>([]);
  const [selected, setSelected] = useState<string>(""); // "beatmap:<key>" OR "upload:<uuid>"
  const [uploading, setUploading] = useState(false);

  // -------------------------
  // Choose MP3 -> Upload/store on backend
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
        // DO NOT set Content-Type manually for FormData in RN fetch
      });

      const data = await resp.json().catch(async () => {
        const raw = await resp.text().catch(() => "");
        throw new Error(
          `Upload failed (${resp.status}) and server did not return JSON. Body: ${raw.slice(
            0,
            200
          )}`
        );
      });

      if (!resp.ok) {
        throw new Error(data.error ?? JSON.stringify(data));
      }

      const newItem: LocalUploadItem = {
        id: data.id,
        name: data.name,
        uri: asset.uri,
        ts: Date.now(),
      };

      setUploads((prev) => [newItem, ...prev]);
      setSelected(`upload:${data.id}`);

      Alert.alert("Uploaded!", `Server ID: ${data.id}\nFile: ${data.name}`);
    } catch (e: any) {
      Alert.alert("Upload error", e?.message ?? String(e));
    } finally {
      setUploading(false);
    }
  }

  // -------------------------
  // Submit -> either hardcoded beatmap OR uploaded mp3
  // -------------------------
  async function submitToBackend() {
    if (!selected) return Alert.alert("Select a song first.");

    const isBeatmap = selected.startsWith("beatmap:");
    const isUpload = selected.startsWith("upload:");

    let url = "";
    if (isBeatmap) {
      const beatmapName = selected.replace("beatmap:", "");
      url = `${BACKEND_BASE}/submit/hardcoded?beatmap=${encodeURIComponent(
        beatmapName
      )}`;
    } else if (isUpload) {
      const uploadId = selected.replace("upload:", "");
      url = `${BACKEND_BASE}/submit/${encodeURIComponent(uploadId)}`;
    } else {
      return Alert.alert("Bad selection value.");
    }

    setUploading(true);
    try {
      const resp = await fetch(url, { method: "POST" });

      const data = await resp.json().catch(async () => {
        const raw = await resp.text().catch(() => "");
        throw new Error(
          `Submit failed (${resp.status}) and server did not return JSON. Body: ${raw.slice(
            0,
            200
          )}`
        );
      });

      if (!resp.ok) {
        throw new Error(data.error ?? JSON.stringify(data));
      }

      Alert.alert("Sent to Pi!", JSON.stringify(data.pi_ack ?? data, null, 2));
    } catch (e: any) {
      Alert.alert("Error", e?.message ?? String(e));
    } finally {
      setUploading(false);
    }
  }

  const selectedLabel = selected.startsWith("beatmap:")
    ? HARDCODED_SONGS.find((s) => `beatmap:${s.key}` === selected)?.label ?? ""
    : selected.startsWith("upload:")
    ? uploads.find((u) => `upload:${u.id}` === selected)?.name ?? ""
    : "";

  return (
    <View style={styles.container}>
      <Text style={styles.title}>MP3 Manager</Text>

      <Pressable style={styles.button} onPress={pickMp3} disabled={uploading}>
        <Text style={styles.buttonText}>Choose MP3</Text>
      </Pressable>

      <Text style={styles.sectionTitle}>Songs</Text>

      <View style={styles.pickerWrap}>
        <Picker
          selectedValue={selected}
          onValueChange={(val) => setSelected(String(val))}
        >
          <Picker.Item label="Select..." value="" />

          {/* Hardcoded songs */}
          {HARDCODED_SONGS.map((s) => (
            <Picker.Item
              key={`beatmap:${s.key}`}
              label={s.label}
              value={`beatmap:${s.key}`}
            />
          ))}

          {/* Uploaded MP3s */}
          {uploads.map((u) => (
            <Picker.Item
              key={`upload:${u.id}`}
              label={u.name}
              value={`upload:${u.id}`}
            />
          ))}
        </Picker>
      </View>

      {selected ? (
        <Text style={styles.note}>Selected: {selectedLabel}</Text>
      ) : (
        <Text style={styles.note}>No selection</Text>
      )}

      <Pressable
        style={[styles.button, (!selected || uploading) && styles.buttonDisabled]}
        onPress={submitToBackend}
        disabled={!selected || uploading}
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