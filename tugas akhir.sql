--
-- PostgreSQL database dump
--

-- Dumped from database version 17.2
-- Dumped by pg_dump version 17.2

-- Started on 2025-02-13 14:57:16

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 222 (class 1259 OID 16449)
-- Name: absensi; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.absensi (
    id_absensi integer NOT NULL,
    id_karyawan integer NOT NULL,
    nama character varying(100) NOT NULL,
    work_type character varying(100) DEFAULT NULL::character varying,
    office character varying(100) DEFAULT NULL::character varying,
    latitude double precision NOT NULL,
    longitude double precision NOT NULL,
    absensi_masuk timestamp without time zone,
    absensi_pulang timestamp without time zone
);


ALTER TABLE public.absensi OWNER TO postgres;

--
-- TOC entry 221 (class 1259 OID 16448)
-- Name: absensi_id_absensi_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.absensi_id_absensi_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.absensi_id_absensi_seq OWNER TO postgres;

--
-- TOC entry 4876 (class 0 OID 0)
-- Dependencies: 221
-- Name: absensi_id_absensi_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.absensi_id_absensi_seq OWNED BY public.absensi.id_absensi;


--
-- TOC entry 218 (class 1259 OID 16427)
-- Name: karyawan; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.karyawan (
    id_karyawan integer NOT NULL,
    nama character varying(100) NOT NULL,
    email character varying(100) NOT NULL,
    password character varying(100) NOT NULL,
    role character varying(5) DEFAULT 'user'::character varying,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT karyawan_role_check CHECK (((role)::text = ANY ((ARRAY['admin'::character varying, 'user'::character varying])::text[])))
);


ALTER TABLE public.karyawan OWNER TO postgres;

--
-- TOC entry 217 (class 1259 OID 16426)
-- Name: karyawan_id_karyawan_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.karyawan_id_karyawan_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.karyawan_id_karyawan_seq OWNER TO postgres;

--
-- TOC entry 4877 (class 0 OID 0)
-- Dependencies: 217
-- Name: karyawan_id_karyawan_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.karyawan_id_karyawan_seq OWNED BY public.karyawan.id_karyawan;


--
-- TOC entry 220 (class 1259 OID 16441)
-- Name: lokasi; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.lokasi (
    id_lokasi integer NOT NULL,
    nama_lokasi character varying(100) NOT NULL,
    latitude numeric(10,8) NOT NULL,
    longitude numeric(11,8) NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.lokasi OWNER TO postgres;

--
-- TOC entry 219 (class 1259 OID 16440)
-- Name: lokasi_id_lokasi_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.lokasi_id_lokasi_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.lokasi_id_lokasi_seq OWNER TO postgres;

--
-- TOC entry 4878 (class 0 OID 0)
-- Dependencies: 219
-- Name: lokasi_id_lokasi_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.lokasi_id_lokasi_seq OWNED BY public.lokasi.id_lokasi;


--
-- TOC entry 4707 (class 2604 OID 16452)
-- Name: absensi id_absensi; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.absensi ALTER COLUMN id_absensi SET DEFAULT nextval('public.absensi_id_absensi_seq'::regclass);


--
-- TOC entry 4702 (class 2604 OID 16430)
-- Name: karyawan id_karyawan; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.karyawan ALTER COLUMN id_karyawan SET DEFAULT nextval('public.karyawan_id_karyawan_seq'::regclass);


--
-- TOC entry 4705 (class 2604 OID 16444)
-- Name: lokasi id_lokasi; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lokasi ALTER COLUMN id_lokasi SET DEFAULT nextval('public.lokasi_id_lokasi_seq'::regclass);


--
-- TOC entry 4870 (class 0 OID 16449)
-- Dependencies: 222
-- Data for Name: absensi; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.absensi (id_absensi, id_karyawan, nama, work_type, office, latitude, longitude, absensi_masuk, absensi_pulang) VALUES (1, 7, 'Bagus', 'WFA', 'Work From Anywhere', -7.28936, 112.799, '2025-01-13 12:55:33', NULL);
INSERT INTO public.absensi (id_absensi, id_karyawan, nama, work_type, office, latitude, longitude, absensi_masuk, absensi_pulang) VALUES (2, 7, 'Bagus', 'WFA', 'Work From Anywhere', -7.28936, 112.799, '2025-01-19 08:38:17', NULL);


--
-- TOC entry 4866 (class 0 OID 16427)
-- Dependencies: 218
-- Data for Name: karyawan; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.karyawan (id_karyawan, nama, email, password, role, created_at) VALUES (1, 'Ronaldo', 'ronaldo@gmail.com', '123', 'user', '2025-02-11 13:37:24.044103');
INSERT INTO public.karyawan (id_karyawan, nama, email, password, role, created_at) VALUES (2, 'coba', 'cintaastutish@gmail.com', '123', 'user', '2025-02-11 13:37:24.044103');
INSERT INTO public.karyawan (id_karyawan, nama, email, password, role, created_at) VALUES (3, 'test', 'elmazulaikay@gmail.com', '123', 'user', '2025-02-11 13:37:24.044103');
INSERT INTO public.karyawan (id_karyawan, nama, email, password, role, created_at) VALUES (4, 'Akbar', 'akbar@gmail.com', '123', 'user', '2025-02-11 13:37:24.044103');
INSERT INTO public.karyawan (id_karyawan, nama, email, password, role, created_at) VALUES (5, 'Imam', 'imam@gmail.com', '123', 'user', '2025-02-11 13:37:24.044103');
INSERT INTO public.karyawan (id_karyawan, nama, email, password, role, created_at) VALUES (6, 'Alan', 'alan@gmail.com', '123', 'user', '2025-02-11 13:37:24.044103');
INSERT INTO public.karyawan (id_karyawan, nama, email, password, role, created_at) VALUES (7, 'Bagus', 'bagus@gmail.com', '123', 'user', '2025-02-11 13:37:24.044103');
INSERT INTO public.karyawan (id_karyawan, nama, email, password, role, created_at) VALUES (8, 'Latif', 'latif@gmail.com', '123', 'user', '2025-02-11 13:37:24.044103');
INSERT INTO public.karyawan (id_karyawan, nama, email, password, role, created_at) VALUES (9, 'Troy', 'troy@gmail.com', '123', 'user', '2025-02-11 13:37:24.044103');


--
-- TOC entry 4868 (class 0 OID 16441)
-- Dependencies: 220
-- Data for Name: lokasi; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.lokasi (id_lokasi, nama_lokasi, latitude, longitude, created_at) VALUES (1, 'Kantor A', -7.77016671, 111.48200720, '2025-02-11 13:37:24.044103');
INSERT INTO public.lokasi (id_lokasi, nama_lokasi, latitude, longitude, created_at) VALUES (2, 'Kantor B', -7.77046782, 111.47960511, '2025-02-11 13:37:24.044103');


--
-- TOC entry 4879 (class 0 OID 0)
-- Dependencies: 221
-- Name: absensi_id_absensi_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.absensi_id_absensi_seq', 2, true);


--
-- TOC entry 4880 (class 0 OID 0)
-- Dependencies: 217
-- Name: karyawan_id_karyawan_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.karyawan_id_karyawan_seq', 17, true);


--
-- TOC entry 4881 (class 0 OID 0)
-- Dependencies: 219
-- Name: lokasi_id_lokasi_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.lokasi_id_lokasi_seq', 2, true);


--
-- TOC entry 4718 (class 2606 OID 16456)
-- Name: absensi absensi_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.absensi
    ADD CONSTRAINT absensi_pkey PRIMARY KEY (id_absensi);


--
-- TOC entry 4712 (class 2606 OID 16439)
-- Name: karyawan karyawan_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.karyawan
    ADD CONSTRAINT karyawan_email_key UNIQUE (email);


--
-- TOC entry 4714 (class 2606 OID 16437)
-- Name: karyawan karyawan_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.karyawan
    ADD CONSTRAINT karyawan_pkey PRIMARY KEY (id_karyawan);


--
-- TOC entry 4716 (class 2606 OID 16447)
-- Name: lokasi lokasi_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lokasi
    ADD CONSTRAINT lokasi_pkey PRIMARY KEY (id_lokasi);


--
-- TOC entry 4719 (class 2606 OID 16457)
-- Name: absensi absensi_id_karyawan_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.absensi
    ADD CONSTRAINT absensi_id_karyawan_fkey FOREIGN KEY (id_karyawan) REFERENCES public.karyawan(id_karyawan);


-- Completed on 2025-02-13 14:57:16

--
-- PostgreSQL database dump complete
--

