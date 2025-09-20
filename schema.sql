--
-- PostgreSQL database dump
--

-- Dumped from database version 15.13 (Debian 15.13-1.pgdg120+1)
-- Dumped by pg_dump version 15.13 (Debian 15.13-1.pgdg120+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: weeklystate; Type: TYPE; Schema: public; Owner: postgres
--

CREATE TYPE public.weeklystate AS ENUM (
    'not_sent',
    'queued',
    'sent',
    'opened',
    'clicked',
    'used',
    'recorded',
    'responded',
    'skipped',
    'expired'
);


ALTER TYPE public.weeklystate OWNER TO postgres;

--
-- Name: weeklytokenstatus; Type: TYPE; Schema: public; Owner: postgres
--

CREATE TYPE public.weeklytokenstatus AS ENUM (
    'active',
    'opened',
    'used',
    'expired'
);


ALTER TYPE public.weeklytokenstatus OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: admin_edit_log; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.admin_edit_log (
    id integer NOT NULL,
    admin_user_id integer,
    target_user_id integer,
    response_id integer,
    action character varying(64) NOT NULL,
    payload jsonb,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.admin_edit_log OWNER TO postgres;

--
-- Name: admin_edit_log_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.admin_edit_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.admin_edit_log_id_seq OWNER TO postgres;

--
-- Name: admin_edit_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.admin_edit_log_id_seq OWNED BY public.admin_edit_log.id;


--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO postgres;

--
-- Name: chapter_compilation; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.chapter_compilation (
    id integer NOT NULL,
    user_id integer NOT NULL,
    chapter character varying NOT NULL,
    version integer NOT NULL,
    status character varying(16) NOT NULL,
    compiled_markdown text NOT NULL,
    gap_questions jsonb,
    used_blocks jsonb,
    model_name character varying(64),
    prompt_tokens integer,
    completion_tokens integer,
    total_tokens integer,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.chapter_compilation OWNER TO postgres;

--
-- Name: chapter_compilation_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.chapter_compilation_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.chapter_compilation_id_seq OWNER TO postgres;

--
-- Name: chapter_compilation_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.chapter_compilation_id_seq OWNED BY public.chapter_compilation.id;


--
-- Name: chapter_meta; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.chapter_meta (
    id integer NOT NULL,
    name character varying NOT NULL,
    display_name character varying NOT NULL,
    "order" integer DEFAULT 0 NOT NULL,
    tint character varying,
    description text,
    keywords text,
    llm_guidance text
);


ALTER TABLE public.chapter_meta OWNER TO postgres;

--
-- Name: chapter_meta_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.chapter_meta_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.chapter_meta_id_seq OWNER TO postgres;

--
-- Name: chapter_meta_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.chapter_meta_id_seq OWNED BY public.chapter_meta.id;


--
-- Name: invite; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.invite (
    id integer NOT NULL,
    email character varying NOT NULL,
    last_sent timestamp with time zone,
    sent_count integer,
    token character varying(64) NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    used_at timestamp with time zone,
    invited_by_user_id integer,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.invite OWNER TO postgres;

--
-- Name: invite_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.invite_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.invite_id_seq OWNER TO postgres;

--
-- Name: invite_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.invite_id_seq OWNED BY public.invite.id;


--
-- Name: kin_group; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.kin_group (
    id integer NOT NULL,
    name character varying(128) NOT NULL,
    kind character varying(32) DEFAULT 'family'::character varying,
    join_code character varying(16),
    created_by integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.kin_group OWNER TO postgres;

--
-- Name: kin_group_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.kin_group_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.kin_group_id_seq OWNER TO postgres;

--
-- Name: kin_group_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.kin_group_id_seq OWNED BY public.kin_group.id;


--
-- Name: kin_membership; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.kin_membership (
    id integer NOT NULL,
    group_id integer NOT NULL,
    user_id integer NOT NULL,
    role character varying(32) DEFAULT 'member'::character varying
);


ALTER TABLE public.kin_membership OWNER TO postgres;

--
-- Name: kin_membership_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.kin_membership_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.kin_membership_id_seq OWNER TO postgres;

--
-- Name: kin_membership_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.kin_membership_id_seq OWNED BY public.kin_membership.id;


--
-- Name: person; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.person (
    id integer NOT NULL,
    owner_user_id integer NOT NULL,
    display_name character varying(128) NOT NULL,
    given_name character varying(64),
    family_name character varying(64),
    birth_year integer,
    death_year integer,
    notes text,
    photo_url character varying(256),
    meta json,
    visibility character varying(16) DEFAULT 'private'::character varying,
    consent_source character varying(32) DEFAULT 'owner'::character varying
);


ALTER TABLE public.person OWNER TO postgres;

--
-- Name: person_alias; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.person_alias (
    id integer NOT NULL,
    person_id integer NOT NULL,
    alias character varying(128) NOT NULL
);


ALTER TABLE public.person_alias OWNER TO postgres;

--
-- Name: person_alias_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.person_alias_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.person_alias_id_seq OWNER TO postgres;

--
-- Name: person_alias_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.person_alias_id_seq OWNED BY public.person_alias.id;


--
-- Name: person_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.person_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.person_id_seq OWNER TO postgres;

--
-- Name: person_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.person_id_seq OWNED BY public.person.id;


--
-- Name: person_share; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.person_share (
    id integer NOT NULL,
    person_id integer NOT NULL,
    group_id integer NOT NULL,
    shared_by_user_id integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.person_share OWNER TO postgres;

--
-- Name: person_share_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.person_share_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.person_share_id_seq OWNER TO postgres;

--
-- Name: person_share_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.person_share_id_seq OWNED BY public.person_share.id;


--
-- Name: prompt; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.prompt (
    id integer NOT NULL,
    text text NOT NULL,
    chapter character varying NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.prompt OWNER TO postgres;

--
-- Name: prompt_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.prompt_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.prompt_id_seq OWNER TO postgres;

--
-- Name: prompt_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.prompt_id_seq OWNED BY public.prompt.id;


--
-- Name: prompt_media; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.prompt_media (
    id integer NOT NULL,
    prompt_id integer,
    file_path character varying NOT NULL,
    media_type character varying NOT NULL,
    thumbnail_url character varying,
    mime_type character varying,
    duration_sec integer,
    sample_rate integer,
    channels integer,
    width integer,
    height integer,
    size_bytes integer,
    codec_audio character varying,
    codec_video character varying,
    wav_path character varying,
    assignee_user_id integer
);


ALTER TABLE public.prompt_media OWNER TO postgres;

--
-- Name: prompt_media_assignee; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.prompt_media_assignee (
    prompt_media_id integer NOT NULL,
    user_id integer NOT NULL
);


ALTER TABLE public.prompt_media_assignee OWNER TO postgres;

--
-- Name: prompt_media_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.prompt_media_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.prompt_media_id_seq OWNER TO postgres;

--
-- Name: prompt_media_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.prompt_media_id_seq OWNED BY public.prompt_media.id;


--
-- Name: prompt_suggestion; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.prompt_suggestion (
    id integer NOT NULL,
    user_id integer NOT NULL,
    prompt_id integer,
    source character varying(16) NOT NULL,
    title character varying(200),
    text text NOT NULL,
    tags json,
    status character varying(16) DEFAULT 'pending'::character varying NOT NULL,
    rationale_json json,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.prompt_suggestion OWNER TO postgres;

--
-- Name: prompt_suggestion_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.prompt_suggestion_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.prompt_suggestion_id_seq OWNER TO postgres;

--
-- Name: prompt_suggestion_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.prompt_suggestion_id_seq OWNED BY public.prompt_suggestion.id;


--
-- Name: prompt_tags; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.prompt_tags (
    prompt_id integer NOT NULL,
    tag_id integer NOT NULL
);


ALTER TABLE public.prompt_tags OWNER TO postgres;

--
-- Name: relationship_edge; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.relationship_edge (
    id integer NOT NULL,
    user_id integer NOT NULL,
    src_id integer NOT NULL,
    dst_id integer NOT NULL,
    rel_type character varying(48) NOT NULL,
    confidence double precision DEFAULT '0.8'::double precision,
    notes text,
    meta json
);


ALTER TABLE public.relationship_edge OWNER TO postgres;

--
-- Name: relationship_edge_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.relationship_edge_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.relationship_edge_id_seq OWNER TO postgres;

--
-- Name: relationship_edge_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.relationship_edge_id_seq OWNED BY public.relationship_edge.id;


--
-- Name: response; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.response (
    id integer NOT NULL,
    prompt_id integer,
    user_id integer,
    response_text text,
    primary_media_url character varying,
    created_at timestamp with time zone DEFAULT now(),
    transcription text,
    title character varying(200),
    ai_polished text,
    ai_polished_at timestamp with time zone,
    primary_thumbnail_path character varying,
    primary_mime_type character varying,
    primary_duration_sec integer,
    primary_sample_rate integer,
    primary_channels integer,
    primary_width integer,
    primary_height integer,
    primary_size_bytes integer,
    primary_codec_audio character varying,
    primary_codec_video character varying,
    primary_wav_path character varying
);


ALTER TABLE public.response OWNER TO postgres;

--
-- Name: response_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.response_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.response_id_seq OWNER TO postgres;

--
-- Name: response_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.response_id_seq OWNED BY public.response.id;


--
-- Name: response_person; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.response_person (
    id integer NOT NULL,
    response_id integer NOT NULL,
    person_id integer NOT NULL,
    alias_used character varying(128),
    start_char integer,
    end_char integer,
    confidence double precision DEFAULT '0.7'::double precision,
    role_hint character varying(48)
);


ALTER TABLE public.response_person OWNER TO postgres;

--
-- Name: response_person_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.response_person_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.response_person_id_seq OWNER TO postgres;

--
-- Name: response_person_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.response_person_id_seq OWNED BY public.response_person.id;


--
-- Name: response_segments; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.response_segments (
    id integer NOT NULL,
    response_id integer NOT NULL,
    order_index integer DEFAULT 0 NOT NULL,
    media_path character varying,
    media_mime character varying,
    transcript text DEFAULT ''::text NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.response_segments OWNER TO postgres;

--
-- Name: response_segments_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.response_segments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.response_segments_id_seq OWNER TO postgres;

--
-- Name: response_segments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.response_segments_id_seq OWNED BY public.response_segments.id;


--
-- Name: response_share; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.response_share (
    id integer NOT NULL,
    token character varying(64) NOT NULL,
    response_id integer NOT NULL,
    user_id integer NOT NULL,
    permanent boolean NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    expires_at timestamp with time zone,
    revoked boolean NOT NULL
);


ALTER TABLE public.response_share OWNER TO postgres;

--
-- Name: response_share_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.response_share_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.response_share_id_seq OWNER TO postgres;

--
-- Name: response_share_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.response_share_id_seq OWNED BY public.response_share.id;


--
-- Name: response_tags; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.response_tags (
    response_id integer NOT NULL,
    tag_id integer NOT NULL
);


ALTER TABLE public.response_tags OWNER TO postgres;

--
-- Name: response_version; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.response_version (
    id integer NOT NULL,
    response_id integer NOT NULL,
    user_id integer NOT NULL,
    title text,
    transcription text,
    tags_json jsonb,
    created_at timestamp with time zone DEFAULT now(),
    edited_by_admin_id integer
);


ALTER TABLE public.response_version OWNER TO postgres;

--
-- Name: response_version_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.response_version_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.response_version_id_seq OWNER TO postgres;

--
-- Name: response_version_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.response_version_id_seq OWNED BY public.response_version.id;


--
-- Name: supporting_media; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.supporting_media (
    id integer NOT NULL,
    response_id integer,
    file_path character varying NOT NULL,
    media_type character varying NOT NULL,
    thumbnail_url character varying,
    mime_type character varying,
    duration_sec integer,
    sample_rate integer,
    channels integer,
    width integer,
    height integer,
    size_bytes integer,
    codec_audio character varying,
    codec_video character varying,
    wav_path character varying
);


ALTER TABLE public.supporting_media OWNER TO postgres;

--
-- Name: supporting_media_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.supporting_media_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.supporting_media_id_seq OWNER TO postgres;

--
-- Name: supporting_media_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.supporting_media_id_seq OWNED BY public.supporting_media.id;


--
-- Name: tag; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.tag (
    id integer NOT NULL,
    name character varying(64) NOT NULL,
    slug character varying(64) NOT NULL,
    color character varying(16),
    CONSTRAINT ck_tag_slug_lower CHECK (((slug)::text = lower((slug)::text)))
);


ALTER TABLE public.tag OWNER TO postgres;

--
-- Name: tag_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.tag_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.tag_id_seq OWNER TO postgres;

--
-- Name: tag_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.tag_id_seq OWNED BY public.tag.id;


--
-- Name: user; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public."user" (
    id integer NOT NULL,
    email character varying NOT NULL,
    username character varying,
    hashed_password character varying NOT NULL,
    is_active boolean,
    is_admin boolean,
    super_admin boolean,
    must_change_password boolean,
    relationship_status character varying,
    goals character varying,
    is_superuser boolean DEFAULT false,
    weekly_current_prompt_id integer,
    weekly_on_deck_prompt_id integer,
    weekly_state public.weeklystate NOT NULL,
    weekly_queued_at timestamp without time zone,
    weekly_sent_at timestamp without time zone,
    weekly_opened_at timestamp without time zone,
    weekly_clicked_at timestamp without time zone,
    weekly_used_at timestamp without time zone,
    weekly_completed_at timestamp without time zone,
    weekly_skipped_at timestamp without time zone,
    weekly_expires_at timestamp without time zone,
    weekly_email_provider_id character varying
);


ALTER TABLE public."user" OWNER TO postgres;

--
-- Name: user_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.user_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_id_seq OWNER TO postgres;

--
-- Name: user_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.user_id_seq OWNED BY public."user".id;


--
-- Name: user_profile; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_profile (
    id integer NOT NULL,
    user_id integer,
    display_name character varying(128),
    birth_year integer,
    location character varying(128),
    relation_roles json,
    interests json,
    accessibility_prefs json,
    consent_flags json,
    bio text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    tag_weights json,
    privacy_prefs json
);


ALTER TABLE public.user_profile OWNER TO postgres;

--
-- Name: user_profile_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.user_profile_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_profile_id_seq OWNER TO postgres;

--
-- Name: user_profile_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.user_profile_id_seq OWNED BY public.user_profile.id;


--
-- Name: user_prompts; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_prompts (
    id integer NOT NULL,
    user_id integer NOT NULL,
    prompt_id integer NOT NULL,
    status character varying DEFAULT 'queued'::character varying,
    score double precision DEFAULT '0'::double precision,
    assigned_at timestamp without time zone DEFAULT now(),
    last_sent_at timestamp without time zone,
    times_sent integer DEFAULT 0
);


ALTER TABLE public.user_prompts OWNER TO postgres;

--
-- Name: user_prompts_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.user_prompts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_prompts_id_seq OWNER TO postgres;

--
-- Name: user_prompts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.user_prompts_id_seq OWNED BY public.user_prompts.id;


--
-- Name: user_weekly_prompts; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_weekly_prompts (
    id integer NOT NULL,
    user_id integer NOT NULL,
    year integer NOT NULL,
    week integer NOT NULL,
    prompt_id integer,
    status character varying(16) DEFAULT 'active'::character varying NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.user_weekly_prompts OWNER TO postgres;

--
-- Name: user_weekly_prompts_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.user_weekly_prompts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_weekly_prompts_id_seq OWNER TO postgres;

--
-- Name: user_weekly_prompts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.user_weekly_prompts_id_seq OWNED BY public.user_weekly_prompts.id;


--
-- Name: user_weekly_skips; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_weekly_skips (
    id integer NOT NULL,
    user_id integer NOT NULL,
    year integer NOT NULL,
    week integer NOT NULL,
    prompt_id integer NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.user_weekly_skips OWNER TO postgres;

--
-- Name: user_weekly_skips_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.user_weekly_skips_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_weekly_skips_id_seq OWNER TO postgres;

--
-- Name: user_weekly_skips_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.user_weekly_skips_id_seq OWNED BY public.user_weekly_skips.id;


--
-- Name: weekly_token; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.weekly_token (
    id integer NOT NULL,
    token character varying(64) NOT NULL,
    user_id integer NOT NULL,
    prompt_id integer NOT NULL,
    status public.weeklytokenstatus NOT NULL,
    sent_at timestamp without time zone,
    opened_at timestamp without time zone,
    clicked_at timestamp without time zone,
    used_at timestamp without time zone,
    completed_at timestamp without time zone,
    expires_at timestamp without time zone
);


ALTER TABLE public.weekly_token OWNER TO postgres;

--
-- Name: weekly_token_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.weekly_token_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.weekly_token_id_seq OWNER TO postgres;

--
-- Name: weekly_token_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.weekly_token_id_seq OWNED BY public.weekly_token.id;


--
-- Name: admin_edit_log id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.admin_edit_log ALTER COLUMN id SET DEFAULT nextval('public.admin_edit_log_id_seq'::regclass);


--
-- Name: chapter_compilation id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chapter_compilation ALTER COLUMN id SET DEFAULT nextval('public.chapter_compilation_id_seq'::regclass);


--
-- Name: chapter_meta id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chapter_meta ALTER COLUMN id SET DEFAULT nextval('public.chapter_meta_id_seq'::regclass);


--
-- Name: invite id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invite ALTER COLUMN id SET DEFAULT nextval('public.invite_id_seq'::regclass);


--
-- Name: kin_group id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kin_group ALTER COLUMN id SET DEFAULT nextval('public.kin_group_id_seq'::regclass);


--
-- Name: kin_membership id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kin_membership ALTER COLUMN id SET DEFAULT nextval('public.kin_membership_id_seq'::regclass);


--
-- Name: person id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.person ALTER COLUMN id SET DEFAULT nextval('public.person_id_seq'::regclass);


--
-- Name: person_alias id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.person_alias ALTER COLUMN id SET DEFAULT nextval('public.person_alias_id_seq'::regclass);


--
-- Name: person_share id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.person_share ALTER COLUMN id SET DEFAULT nextval('public.person_share_id_seq'::regclass);


--
-- Name: prompt id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt ALTER COLUMN id SET DEFAULT nextval('public.prompt_id_seq'::regclass);


--
-- Name: prompt_media id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_media ALTER COLUMN id SET DEFAULT nextval('public.prompt_media_id_seq'::regclass);


--
-- Name: prompt_suggestion id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_suggestion ALTER COLUMN id SET DEFAULT nextval('public.prompt_suggestion_id_seq'::regclass);


--
-- Name: relationship_edge id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.relationship_edge ALTER COLUMN id SET DEFAULT nextval('public.relationship_edge_id_seq'::regclass);


--
-- Name: response id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response ALTER COLUMN id SET DEFAULT nextval('public.response_id_seq'::regclass);


--
-- Name: response_person id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_person ALTER COLUMN id SET DEFAULT nextval('public.response_person_id_seq'::regclass);


--
-- Name: response_segments id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_segments ALTER COLUMN id SET DEFAULT nextval('public.response_segments_id_seq'::regclass);


--
-- Name: response_share id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_share ALTER COLUMN id SET DEFAULT nextval('public.response_share_id_seq'::regclass);


--
-- Name: response_version id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_version ALTER COLUMN id SET DEFAULT nextval('public.response_version_id_seq'::regclass);


--
-- Name: supporting_media id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.supporting_media ALTER COLUMN id SET DEFAULT nextval('public.supporting_media_id_seq'::regclass);


--
-- Name: tag id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tag ALTER COLUMN id SET DEFAULT nextval('public.tag_id_seq'::regclass);


--
-- Name: user id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public."user" ALTER COLUMN id SET DEFAULT nextval('public.user_id_seq'::regclass);


--
-- Name: user_profile id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_profile ALTER COLUMN id SET DEFAULT nextval('public.user_profile_id_seq'::regclass);


--
-- Name: user_prompts id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_prompts ALTER COLUMN id SET DEFAULT nextval('public.user_prompts_id_seq'::regclass);


--
-- Name: user_weekly_prompts id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_weekly_prompts ALTER COLUMN id SET DEFAULT nextval('public.user_weekly_prompts_id_seq'::regclass);


--
-- Name: user_weekly_skips id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_weekly_skips ALTER COLUMN id SET DEFAULT nextval('public.user_weekly_skips_id_seq'::regclass);


--
-- Name: weekly_token id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.weekly_token ALTER COLUMN id SET DEFAULT nextval('public.weekly_token_id_seq'::regclass);


--
-- Name: admin_edit_log admin_edit_log_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.admin_edit_log
    ADD CONSTRAINT admin_edit_log_pkey PRIMARY KEY (id);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: chapter_compilation chapter_compilation_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chapter_compilation
    ADD CONSTRAINT chapter_compilation_pkey PRIMARY KEY (id);


--
-- Name: chapter_meta chapter_meta_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chapter_meta
    ADD CONSTRAINT chapter_meta_pkey PRIMARY KEY (id);


--
-- Name: invite invite_email_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invite
    ADD CONSTRAINT invite_email_key UNIQUE (email);


--
-- Name: invite invite_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invite
    ADD CONSTRAINT invite_pkey PRIMARY KEY (id);


--
-- Name: kin_group kin_group_join_code_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kin_group
    ADD CONSTRAINT kin_group_join_code_key UNIQUE (join_code);


--
-- Name: kin_group kin_group_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kin_group
    ADD CONSTRAINT kin_group_pkey PRIMARY KEY (id);


--
-- Name: kin_membership kin_membership_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kin_membership
    ADD CONSTRAINT kin_membership_pkey PRIMARY KEY (id);


--
-- Name: person_alias person_alias_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.person_alias
    ADD CONSTRAINT person_alias_pkey PRIMARY KEY (id);


--
-- Name: person person_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.person
    ADD CONSTRAINT person_pkey PRIMARY KEY (id);


--
-- Name: person_share person_share_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.person_share
    ADD CONSTRAINT person_share_pkey PRIMARY KEY (id);


--
-- Name: prompt_media prompt_media_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_media
    ADD CONSTRAINT prompt_media_pkey PRIMARY KEY (id);


--
-- Name: prompt prompt_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt
    ADD CONSTRAINT prompt_pkey PRIMARY KEY (id);


--
-- Name: prompt_suggestion prompt_suggestion_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_suggestion
    ADD CONSTRAINT prompt_suggestion_pkey PRIMARY KEY (id);


--
-- Name: relationship_edge relationship_edge_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.relationship_edge
    ADD CONSTRAINT relationship_edge_pkey PRIMARY KEY (id);


--
-- Name: response_person response_person_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_person
    ADD CONSTRAINT response_person_pkey PRIMARY KEY (id);


--
-- Name: response response_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response
    ADD CONSTRAINT response_pkey PRIMARY KEY (id);


--
-- Name: response_segments response_segments_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_segments
    ADD CONSTRAINT response_segments_pkey PRIMARY KEY (id);


--
-- Name: response_share response_share_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_share
    ADD CONSTRAINT response_share_pkey PRIMARY KEY (id);


--
-- Name: response_version response_version_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_version
    ADD CONSTRAINT response_version_pkey PRIMARY KEY (id);


--
-- Name: supporting_media supporting_media_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.supporting_media
    ADD CONSTRAINT supporting_media_pkey PRIMARY KEY (id);


--
-- Name: tag tag_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tag
    ADD CONSTRAINT tag_name_key UNIQUE (name);


--
-- Name: tag tag_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tag
    ADD CONSTRAINT tag_pkey PRIMARY KEY (id);


--
-- Name: chapter_compilation uq_compilation_user_chapter_version; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chapter_compilation
    ADD CONSTRAINT uq_compilation_user_chapter_version UNIQUE (user_id, chapter, version);


--
-- Name: kin_membership uq_kin_group_user; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kin_membership
    ADD CONSTRAINT uq_kin_group_user UNIQUE (group_id, user_id);


--
-- Name: prompt_media_assignee uq_media_user_once; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_media_assignee
    ADD CONSTRAINT uq_media_user_once PRIMARY KEY (prompt_media_id, user_id);


--
-- Name: person_share uq_person_group_once; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.person_share
    ADD CONSTRAINT uq_person_group_once UNIQUE (person_id, group_id);


--
-- Name: prompt_tags uq_prompt_tag; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_tags
    ADD CONSTRAINT uq_prompt_tag PRIMARY KEY (prompt_id, tag_id);


--
-- Name: relationship_edge uq_rel_once; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.relationship_edge
    ADD CONSTRAINT uq_rel_once UNIQUE (user_id, src_id, dst_id, rel_type);


--
-- Name: response_tags uq_response_tag; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_tags
    ADD CONSTRAINT uq_response_tag PRIMARY KEY (response_id, tag_id);


--
-- Name: user_prompts uq_user_prompt; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_prompts
    ADD CONSTRAINT uq_user_prompt UNIQUE (user_id, prompt_id);


--
-- Name: user_weekly_prompts uq_user_week; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_weekly_prompts
    ADD CONSTRAINT uq_user_week UNIQUE (user_id, year, week);


--
-- Name: user_weekly_skips uq_user_week_prompt_skip; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_weekly_skips
    ADD CONSTRAINT uq_user_week_prompt_skip UNIQUE (user_id, year, week, prompt_id);


--
-- Name: weekly_token uq_weeklytoken_user_prompt; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.weekly_token
    ADD CONSTRAINT uq_weeklytoken_user_prompt UNIQUE (user_id, prompt_id);


--
-- Name: user user_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public."user"
    ADD CONSTRAINT user_pkey PRIMARY KEY (id);


--
-- Name: user_profile user_profile_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_profile
    ADD CONSTRAINT user_profile_pkey PRIMARY KEY (id);


--
-- Name: user_prompts user_prompts_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_prompts
    ADD CONSTRAINT user_prompts_pkey PRIMARY KEY (id);


--
-- Name: user_weekly_prompts user_weekly_prompts_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_weekly_prompts
    ADD CONSTRAINT user_weekly_prompts_pkey PRIMARY KEY (id);


--
-- Name: user_weekly_skips user_weekly_skips_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_weekly_skips
    ADD CONSTRAINT user_weekly_skips_pkey PRIMARY KEY (id);


--
-- Name: weekly_token weekly_token_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.weekly_token
    ADD CONSTRAINT weekly_token_pkey PRIMARY KEY (id);


--
-- Name: ix_admin_edit_log_admin_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_admin_edit_log_admin_user_id ON public.admin_edit_log USING btree (admin_user_id);


--
-- Name: ix_admin_edit_log_response_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_admin_edit_log_response_id ON public.admin_edit_log USING btree (response_id);


--
-- Name: ix_admin_edit_log_target_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_admin_edit_log_target_user_id ON public.admin_edit_log USING btree (target_user_id);


--
-- Name: ix_chapter_compilation_chapter; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_chapter_compilation_chapter ON public.chapter_compilation USING btree (chapter);


--
-- Name: ix_chapter_compilation_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_chapter_compilation_user_id ON public.chapter_compilation USING btree (user_id);


--
-- Name: ix_chapter_meta_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_chapter_meta_id ON public.chapter_meta USING btree (id);


--
-- Name: ix_chapter_meta_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_chapter_meta_name ON public.chapter_meta USING btree (name);


--
-- Name: ix_compilation_user_chapter_created; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_compilation_user_chapter_created ON public.chapter_compilation USING btree (user_id, chapter, created_at);


--
-- Name: ix_invite_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_invite_id ON public.invite USING btree (id);


--
-- Name: ix_invite_token; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_invite_token ON public.invite USING btree (token);


--
-- Name: ix_kin_membership_group_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_kin_membership_group_id ON public.kin_membership USING btree (group_id);


--
-- Name: ix_kin_membership_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_kin_membership_user_id ON public.kin_membership USING btree (user_id);


--
-- Name: ix_person_alias_alias; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_person_alias_alias ON public.person_alias USING btree (alias);


--
-- Name: ix_person_alias_person_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_person_alias_person_id ON public.person_alias USING btree (person_id);


--
-- Name: ix_person_owner_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_person_owner_user_id ON public.person USING btree (owner_user_id);


--
-- Name: ix_person_share_group_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_person_share_group_id ON public.person_share USING btree (group_id);


--
-- Name: ix_person_share_person_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_person_share_person_id ON public.person_share USING btree (person_id);


--
-- Name: ix_prompt_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_prompt_id ON public.prompt USING btree (id);


--
-- Name: ix_prompt_media_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_prompt_media_id ON public.prompt_media USING btree (id);


--
-- Name: ix_prompt_suggestion_status; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_prompt_suggestion_status ON public.prompt_suggestion USING btree (status);


--
-- Name: ix_prompt_suggestion_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_prompt_suggestion_user_id ON public.prompt_suggestion USING btree (user_id);


--
-- Name: ix_relationship_edge_dst_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_relationship_edge_dst_id ON public.relationship_edge USING btree (dst_id);


--
-- Name: ix_relationship_edge_rel_type; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_relationship_edge_rel_type ON public.relationship_edge USING btree (rel_type);


--
-- Name: ix_relationship_edge_src_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_relationship_edge_src_id ON public.relationship_edge USING btree (src_id);


--
-- Name: ix_relationship_edge_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_relationship_edge_user_id ON public.relationship_edge USING btree (user_id);


--
-- Name: ix_response_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_response_id ON public.response USING btree (id);


--
-- Name: ix_response_person_person_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_response_person_person_id ON public.response_person USING btree (person_id);


--
-- Name: ix_response_person_response_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_response_person_response_id ON public.response_person USING btree (response_id);


--
-- Name: ix_response_segments_response_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_response_segments_response_id ON public.response_segments USING btree (response_id);


--
-- Name: ix_response_share_response_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_response_share_response_id ON public.response_share USING btree (response_id);


--
-- Name: ix_response_share_token; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_response_share_token ON public.response_share USING btree (token);


--
-- Name: ix_response_share_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_response_share_user_id ON public.response_share USING btree (user_id);


--
-- Name: ix_response_title; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_response_title ON public.response USING btree (title);


--
-- Name: ix_response_version_response_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_response_version_response_id ON public.response_version USING btree (response_id);


--
-- Name: ix_response_version_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_response_version_user_id ON public.response_version USING btree (user_id);


--
-- Name: ix_supporting_media_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_supporting_media_id ON public.supporting_media USING btree (id);


--
-- Name: ix_tag_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_tag_id ON public.tag USING btree (id);


--
-- Name: ix_tag_slug; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_tag_slug ON public.tag USING btree (slug);


--
-- Name: ix_user_email; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_user_email ON public."user" USING btree (email);


--
-- Name: ix_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_user_id ON public."user" USING btree (id);


--
-- Name: ix_user_profile_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_user_profile_user_id ON public.user_profile USING btree (user_id);


--
-- Name: ix_user_prompts_prompt_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_user_prompts_prompt_id ON public.user_prompts USING btree (prompt_id);


--
-- Name: ix_user_prompts_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_user_prompts_user_id ON public.user_prompts USING btree (user_id);


--
-- Name: ix_user_username; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_user_username ON public."user" USING btree (username);


--
-- Name: ix_user_weekly_current_prompt_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_user_weekly_current_prompt_id ON public."user" USING btree (weekly_current_prompt_id);


--
-- Name: ix_user_weekly_on_deck_prompt_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_user_weekly_on_deck_prompt_id ON public."user" USING btree (weekly_on_deck_prompt_id);


--
-- Name: ix_uwp_user_year_week; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_uwp_user_year_week ON public.user_weekly_prompts USING btree (user_id, year, week);


--
-- Name: ix_uws_user_year_week_prompt; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_uws_user_year_week_prompt ON public.user_weekly_skips USING btree (user_id, year, week, prompt_id);


--
-- Name: ix_weekly_token_prompt_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_weekly_token_prompt_id ON public.weekly_token USING btree (prompt_id);


--
-- Name: ix_weekly_token_token; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_weekly_token_token ON public.weekly_token USING btree (token);


--
-- Name: ix_weekly_token_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_weekly_token_user_id ON public.weekly_token USING btree (user_id);


--
-- Name: uq_tag_name_ci; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX uq_tag_name_ci ON public.tag USING btree (lower((name)::text));


--
-- Name: uq_tag_slug; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX uq_tag_slug ON public.tag USING btree (slug);


--
-- Name: admin_edit_log admin_edit_log_admin_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.admin_edit_log
    ADD CONSTRAINT admin_edit_log_admin_user_id_fkey FOREIGN KEY (admin_user_id) REFERENCES public."user"(id) ON DELETE SET NULL;


--
-- Name: admin_edit_log admin_edit_log_response_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.admin_edit_log
    ADD CONSTRAINT admin_edit_log_response_id_fkey FOREIGN KEY (response_id) REFERENCES public.response(id) ON DELETE SET NULL;


--
-- Name: admin_edit_log admin_edit_log_target_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.admin_edit_log
    ADD CONSTRAINT admin_edit_log_target_user_id_fkey FOREIGN KEY (target_user_id) REFERENCES public."user"(id) ON DELETE SET NULL;


--
-- Name: chapter_compilation chapter_compilation_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chapter_compilation
    ADD CONSTRAINT chapter_compilation_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- Name: prompt_media fk_prompt_media_user; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_media
    ADD CONSTRAINT fk_prompt_media_user FOREIGN KEY (assignee_user_id) REFERENCES public."user"(id) ON DELETE SET NULL;


--
-- Name: user fk_user_weekly_current_prompt; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public."user"
    ADD CONSTRAINT fk_user_weekly_current_prompt FOREIGN KEY (weekly_current_prompt_id) REFERENCES public.prompt(id) ON DELETE SET NULL;


--
-- Name: user fk_user_weekly_on_deck_prompt; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public."user"
    ADD CONSTRAINT fk_user_weekly_on_deck_prompt FOREIGN KEY (weekly_on_deck_prompt_id) REFERENCES public.prompt(id) ON DELETE SET NULL;


--
-- Name: invite invite_invited_by_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.invite
    ADD CONSTRAINT invite_invited_by_user_id_fkey FOREIGN KEY (invited_by_user_id) REFERENCES public."user"(id) ON DELETE SET NULL;


--
-- Name: kin_group kin_group_created_by_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kin_group
    ADD CONSTRAINT kin_group_created_by_fkey FOREIGN KEY (created_by) REFERENCES public."user"(id) ON DELETE SET NULL;


--
-- Name: kin_membership kin_membership_group_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kin_membership
    ADD CONSTRAINT kin_membership_group_id_fkey FOREIGN KEY (group_id) REFERENCES public.kin_group(id) ON DELETE CASCADE;


--
-- Name: kin_membership kin_membership_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kin_membership
    ADD CONSTRAINT kin_membership_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- Name: person_alias person_alias_person_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.person_alias
    ADD CONSTRAINT person_alias_person_id_fkey FOREIGN KEY (person_id) REFERENCES public.person(id) ON DELETE CASCADE;


--
-- Name: person person_owner_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.person
    ADD CONSTRAINT person_owner_user_id_fkey FOREIGN KEY (owner_user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- Name: person_share person_share_group_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.person_share
    ADD CONSTRAINT person_share_group_id_fkey FOREIGN KEY (group_id) REFERENCES public.kin_group(id) ON DELETE CASCADE;


--
-- Name: person_share person_share_person_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.person_share
    ADD CONSTRAINT person_share_person_id_fkey FOREIGN KEY (person_id) REFERENCES public.person(id) ON DELETE CASCADE;


--
-- Name: person_share person_share_shared_by_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.person_share
    ADD CONSTRAINT person_share_shared_by_user_id_fkey FOREIGN KEY (shared_by_user_id) REFERENCES public."user"(id) ON DELETE SET NULL;


--
-- Name: prompt_media_assignee prompt_media_assignee_prompt_media_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_media_assignee
    ADD CONSTRAINT prompt_media_assignee_prompt_media_id_fkey FOREIGN KEY (prompt_media_id) REFERENCES public.prompt_media(id) ON DELETE CASCADE;


--
-- Name: prompt_media_assignee prompt_media_assignee_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_media_assignee
    ADD CONSTRAINT prompt_media_assignee_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- Name: prompt_media prompt_media_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_media
    ADD CONSTRAINT prompt_media_prompt_id_fkey FOREIGN KEY (prompt_id) REFERENCES public.prompt(id) ON DELETE CASCADE;


--
-- Name: prompt_suggestion prompt_suggestion_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_suggestion
    ADD CONSTRAINT prompt_suggestion_prompt_id_fkey FOREIGN KEY (prompt_id) REFERENCES public.prompt(id) ON DELETE SET NULL;


--
-- Name: prompt_suggestion prompt_suggestion_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_suggestion
    ADD CONSTRAINT prompt_suggestion_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- Name: prompt_tags prompt_tags_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_tags
    ADD CONSTRAINT prompt_tags_prompt_id_fkey FOREIGN KEY (prompt_id) REFERENCES public.prompt(id) ON DELETE CASCADE;


--
-- Name: prompt_tags prompt_tags_tag_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.prompt_tags
    ADD CONSTRAINT prompt_tags_tag_id_fkey FOREIGN KEY (tag_id) REFERENCES public.tag(id) ON DELETE CASCADE;


--
-- Name: relationship_edge relationship_edge_dst_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.relationship_edge
    ADD CONSTRAINT relationship_edge_dst_id_fkey FOREIGN KEY (dst_id) REFERENCES public.person(id) ON DELETE CASCADE;


--
-- Name: relationship_edge relationship_edge_src_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.relationship_edge
    ADD CONSTRAINT relationship_edge_src_id_fkey FOREIGN KEY (src_id) REFERENCES public.person(id) ON DELETE CASCADE;


--
-- Name: relationship_edge relationship_edge_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.relationship_edge
    ADD CONSTRAINT relationship_edge_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- Name: response_person response_person_person_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_person
    ADD CONSTRAINT response_person_person_id_fkey FOREIGN KEY (person_id) REFERENCES public.person(id) ON DELETE CASCADE;


--
-- Name: response_person response_person_response_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_person
    ADD CONSTRAINT response_person_response_id_fkey FOREIGN KEY (response_id) REFERENCES public.response(id) ON DELETE CASCADE;


--
-- Name: response response_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response
    ADD CONSTRAINT response_prompt_id_fkey FOREIGN KEY (prompt_id) REFERENCES public.prompt(id) ON DELETE CASCADE;


--
-- Name: response_segments response_segments_response_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_segments
    ADD CONSTRAINT response_segments_response_id_fkey FOREIGN KEY (response_id) REFERENCES public.response(id) ON DELETE CASCADE;


--
-- Name: response_share response_share_response_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_share
    ADD CONSTRAINT response_share_response_id_fkey FOREIGN KEY (response_id) REFERENCES public.response(id) ON DELETE CASCADE;


--
-- Name: response_share response_share_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_share
    ADD CONSTRAINT response_share_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- Name: response_tags response_tags_response_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_tags
    ADD CONSTRAINT response_tags_response_id_fkey FOREIGN KEY (response_id) REFERENCES public.response(id) ON DELETE CASCADE;


--
-- Name: response_tags response_tags_tag_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_tags
    ADD CONSTRAINT response_tags_tag_id_fkey FOREIGN KEY (tag_id) REFERENCES public.tag(id) ON DELETE CASCADE;


--
-- Name: response response_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response
    ADD CONSTRAINT response_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- Name: response_version response_version_edited_by_admin_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_version
    ADD CONSTRAINT response_version_edited_by_admin_id_fkey FOREIGN KEY (edited_by_admin_id) REFERENCES public."user"(id) ON DELETE SET NULL;


--
-- Name: response_version response_version_response_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_version
    ADD CONSTRAINT response_version_response_id_fkey FOREIGN KEY (response_id) REFERENCES public.response(id) ON DELETE CASCADE;


--
-- Name: response_version response_version_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.response_version
    ADD CONSTRAINT response_version_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- Name: supporting_media supporting_media_response_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.supporting_media
    ADD CONSTRAINT supporting_media_response_id_fkey FOREIGN KEY (response_id) REFERENCES public.response(id) ON DELETE CASCADE;


--
-- Name: user_profile user_profile_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_profile
    ADD CONSTRAINT user_profile_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- Name: user_prompts user_prompts_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_prompts
    ADD CONSTRAINT user_prompts_prompt_id_fkey FOREIGN KEY (prompt_id) REFERENCES public.prompt(id) ON DELETE CASCADE;


--
-- Name: user_prompts user_prompts_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_prompts
    ADD CONSTRAINT user_prompts_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- Name: user_weekly_prompts user_weekly_prompts_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_weekly_prompts
    ADD CONSTRAINT user_weekly_prompts_prompt_id_fkey FOREIGN KEY (prompt_id) REFERENCES public.prompt(id) ON DELETE SET NULL;


--
-- Name: user_weekly_prompts user_weekly_prompts_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_weekly_prompts
    ADD CONSTRAINT user_weekly_prompts_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- Name: user_weekly_skips user_weekly_skips_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_weekly_skips
    ADD CONSTRAINT user_weekly_skips_prompt_id_fkey FOREIGN KEY (prompt_id) REFERENCES public.prompt(id) ON DELETE CASCADE;


--
-- Name: user_weekly_skips user_weekly_skips_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_weekly_skips
    ADD CONSTRAINT user_weekly_skips_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- Name: weekly_token weekly_token_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.weekly_token
    ADD CONSTRAINT weekly_token_prompt_id_fkey FOREIGN KEY (prompt_id) REFERENCES public.prompt(id) ON DELETE CASCADE;


--
-- Name: weekly_token weekly_token_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.weekly_token
    ADD CONSTRAINT weekly_token_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

