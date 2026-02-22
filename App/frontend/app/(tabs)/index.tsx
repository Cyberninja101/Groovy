import React, { useState } from "react";
import {
  View,
  Text,
  Pressable,
  StyleSheet,
  ActivityIndicator,
  Alert,
  ScrollView,
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

const HARDCODED_SONGS = [
  { label: "stacyMom", key: "stacyMom" },
  { label: "seven nation army", key: "seven_nation_army" },
  { label: "APT", key: "APT" },
];

const NOTES_AT_A_TIME_OPTIONS = [
  { label: "0.25", value: 0.25 },
  { label: "0.5", value: 0.5 },
  { label: "0.75", value: 0.75 },
  { label: "1", value: 1 },
  { label: "1.25", value: 1.25 },
  { label: "1.5", value: 1.5 },
  { label: "one note at a time", value: -1 },
];

export default function HomeScreen() {
  const [uploads, setUploads] = useState<LocalUploadItem[]>([]);
  const [selected, setSelected] = useState<string>(""); // "beatmap:<key>" OR "upload:<uuid>"
  const [uploading, setUploading] = useState(false);

  // speed dropdown (ONLY used for uploaded MP3 submits)
  const [notesAtATime, setNotesAtATime] = useState<number>(1);

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

      if (!resp.ok) throw new Error(data.error ?? JSON.stringify(data));

      const newItem: LocalUploadItem = {
        id: data.id,
        name: data.name,
        uri: asset.uri,
        ts: Date.now(),
      };

      setUploads((prev) => [newItem, ...prev]);
      setSelected(`upload:${data.id}`);

      Alert.alert("Uploaded!", data.name);
    } catch (e: any) {
      Alert.alert("Upload error", e?.message ?? String(e));
    } finally {
      setUploading(false);
    }
  }

  async function submit() {
    if (!selected) return Alert.alert("Select a song first.");

    const isBeatmap = selected.startsWith("beatmap:");
    const isUpload = selected.startsWith("upload:");

    let url = "";

    if (isBeatmap) {
      // ✅ HARD-CODED: IGNORE speed (do NOT send notes_at_a_time)
      const beatmapName = selected.replace("beatmap:", "");
      url = `${BACKEND_BASE}/submit/hardcoded?beatmap=${encodeURIComponent(
        beatmapName
      )}`;
    } else if (isUpload) {
      // ✅ UPLOAD: include speed
      const uploadId = selected.replace("upload:", "");
      url = `${BACKEND_BASE}/submit/${encodeURIComponent(
        uploadId
      )}?notes_at_a_time=${encodeURIComponent(String(notesAtATime))}`;
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

      if (!resp.ok) throw new Error(data.error ?? JSON.stringify(data));

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

  const isBeatmap = selected.startsWith("beatmap:");

  return (
    <ScrollView
      contentContainerStyle={styles.container}
      keyboardShouldPersistTaps="handled"
    >
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
          {HARDCODED_SONGS.map((s) => (
            <Picker.Item
              key={`beatmap:${s.key}`}
              label={s.label}
              value={`beatmap:${s.key}`}
            />
          ))}
          {uploads.map((u) => (
            <Picker.Item
              key={`upload:${u.id}`}
              label={u.name}
              value={`upload:${u.id}`}
            />
          ))}
        </Picker>
      </View>

      {/* Speed dropdown (disabled + visually dimmed for hardcoded) */}
      <Text style={styles.sectionTitle}>Speed (uploads only)</Text>
      <View style={[styles.pickerWrap, isBeatmap && styles.pickerDisabled]}>
        <Picker
          enabled={!isBeatmap}
          selectedValue={notesAtATime}
          onValueChange={(val) => setNotesAtATime(Number(val))}
        >
          {NOTES_AT_A_TIME_OPTIONS.map((opt) => (
            <Picker.Item
              key={String(opt.value)}
              label={opt.label}
              value={opt.value}
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
        onPress={submit}
        disabled={!selected || uploading}
      >
        {uploading ? (
          <ActivityIndicator color="white" />
        ) : (
          <Text style={styles.buttonText}>Submit</Text>
        )}
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    gap: 14,
    paddingBottom: 40, // extra space for scrolling
  },
  title: { fontSize: 28, fontWeight: "700", textAlign: "center", marginTop: 40 },
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
  pickerDisabled: {
    opacity: 0.5,
  },
  note: { marginTop: 6, textAlign: "center", opacity: 0.8 },
});