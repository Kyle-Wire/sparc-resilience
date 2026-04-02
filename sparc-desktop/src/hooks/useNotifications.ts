import { createContext, useContext, useCallback, useState } from "react";

export type NotificationType = "success" | "error" | "warning" | "info";

export interface Notification {
  id: number;
  type: NotificationType;
  message: string;
}

interface NotificationContextValue {
  notifications: Notification[];
  notify: (type: NotificationType, message: string) => void;
  dismiss: (id: number) => void;
}

let _nextId = 0;

export const NotificationContext = createContext<NotificationContextValue>({
  notifications: [],
  notify: () => {},
  dismiss: () => {},
});

export function useNotification() {
  return useContext(NotificationContext);
}

export function useNotificationState() {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const dismiss = useCallback((id: number) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const notify = useCallback(
    (type: NotificationType, message: string) => {
      const id = ++_nextId;
      setNotifications((prev) => [...prev, { id, type, message }]);
      const delay = type === "error" ? 6000 : 3000;
      setTimeout(() => dismiss(id), delay);
    },
    [dismiss],
  );

  return { notifications, notify, dismiss };
}
