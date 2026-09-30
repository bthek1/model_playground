import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { getMe, login, register } from '../api/auth'
import { queryKeys } from '../api/queryKeys'
import { BACKEND_ENABLED } from '../lib/features'
import type { LoginPayload, RegisterPayload } from '../types/auth'

export function useMe() {
  return useQuery({
    queryKey: queryKeys.auth.me,
    queryFn: getMe,
    // A static build has no /api to ask (#57); a stale token must not make it try.
    enabled: BACKEND_ENABLED && !!localStorage.getItem('access_token'),
    retry: false,
  })
}

export function useLogin() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (payload: LoginPayload) => login(payload),
    onSuccess: (tokens) => {
      localStorage.setItem('access_token', tokens.access)
      localStorage.setItem('refresh_token', tokens.refresh)
      queryClient.invalidateQueries({ queryKey: queryKeys.auth.me })
    },
  })
}

export function useRegister() {
  return useMutation({
    mutationFn: (payload: RegisterPayload) => register(payload),
  })
}

export function useLogout() {
  const queryClient = useQueryClient()
  return () => {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    queryClient.clear()
  }
}
